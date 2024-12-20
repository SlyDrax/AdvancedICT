from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import laspy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pointnet.model import PointNetDenseCls
from tqdm import tqdm  # For progress bars
from block_test import create_blocks_inference, InferenceBlockDataset, run_inference

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def remap_labels(labels, label_mapping):
    remapped_labels = np.full_like(labels, fill_value=-1, dtype=np.int32)
    for original_label, mapped_label in label_mapping.items():
        remapped_labels[labels == original_label] = mapped_label
    return remapped_labels

def compute_metrics(global_labels, ground_truth, num_classes):
    mask = ground_truth != -1  # Ignore points with invalid remapped labels
    ground_truth = ground_truth[mask]
    global_labels = global_labels[mask]

    overall_accuracy = accuracy_score(ground_truth, global_labels)
    confusion = confusion_matrix(ground_truth, global_labels, labels=range(num_classes))
    
    class_iou = []
    for i in range(num_classes):
        intersection = confusion[i, i]
        union = confusion[i, :].sum() + confusion[:, i].sum() - intersection
        if union > 0:
            class_iou.append(intersection / union)
        else:
            class_iou.append(0.0)
    
    mean_iou = np.mean(class_iou)
    return overall_accuracy, confusion, class_iou, mean_iou

def evaluate_building_accuracy(global_labels, ground_truth, building_label=1):
    # Filter for building class only
    building_mask = (ground_truth == building_label)
    building_truth = ground_truth[building_mask]
    building_preds = global_labels[building_mask]

    # If no building points are present in the ground truth, return zeroed metrics
    if len(building_truth) == 0:
        print("No building points in ground truth.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "IoU": 0.0}

    # Calculate metrics
    accuracy = accuracy_score(building_truth, building_preds)
    precision = precision_score(building_truth, building_preds, average='binary', pos_label=building_label)
    recall = recall_score(building_truth, building_preds, average='binary', pos_label=building_label)

    # IoU calculation
    conf_matrix = confusion_matrix(building_truth, building_preds, labels=[building_label])
    intersection = conf_matrix[0, 0]  # True Positives
    union = intersection + (conf_matrix.sum() - intersection)  # TP + FP + FN
    iou = intersection / union if union > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "IoU": iou,
    }


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Block Inference")
    parser.add_argument("--input_las", type=str, required=True, help="Path to new .las file for inference")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Trained model .pth")
    parser.add_argument("--output_las", type=str, default="inferred.las", help="Output .las with predicted classes")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--block_size", type=float, default=50.0, help="Size of each block in meters")
    parser.add_argument("--stride", type=float, default=25.0, help="Stride for block sampling in meters")
    parser.add_argument("--use_intensity", action="store_true", help="Use intensity if available")
    parser.add_argument("--label_mapping", type=str, help="JSON file containing label mapping (original -> model labels)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label mapping
    if args.label_mapping:
        import json
        with open(args.label_mapping, 'r') as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {
            1: 0,  # Terrain
            2: 0,  # Terrain
            3: 0,  # Terrain
            4: 0,  # Terrain
            5: 0,  # Terrain
            6: 0,  # Terrain
            7: 0,  # grimstad building
            8: 1,  # dales building
        }

    print(f"Using label mapping: {label_mapping}")

    # 1. Read LAS
    print(f"Reading LAS file: {args.input_las}")
    las = laspy.read(args.input_las)
    x, y, z = las.x, las.y, las.z
    ground_truth = las.classification  # Ground truth labels

    # Remap labels to match model's label set
    ground_truth = remap_labels(ground_truth, label_mapping)

    if args.use_intensity and hasattr(las, 'intensity'):
        intensity = las.intensity
        points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
        print("Using intensity data.")
    else:
        points = np.stack([x, y, z], axis=-1).astype(np.float32)
        print("Intensity data not used.")

    # 2. Partition new .las into blocks, local normalize
    print("Partitioning point cloud into blocks...")
    blocks, block_idxs = create_blocks_inference(points,
                                                 block_size=args.block_size,
                                                 stride=args.stride)

    if len(blocks) == 0:
        print("No blocks found. Check your block_size/stride or .las content.")
        return

    print(f"Total blocks created: {len(blocks)}")

    # 3. Create Dataset
    inference_dataset = InferenceBlockDataset(blocks)

    # 4. Load model
    print(f"Loading model from: {args.model_ckpt}")
    model = PointNetDenseCls(k=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # 5. Run inference on each block
    print("Running inference on blocks...")
    block_preds_list = run_inference(model, inference_dataset,
                                     1, args.num_classes, device)
    print("Inference completed.")

    # 6. Stitch predictions back to a global array using majority voting
    print("Stitching predictions back to the global point cloud using majority voting...")
    num_points = points.shape[0]
    num_classes = args.num_classes
    class_counts = np.zeros((num_points, num_classes), dtype=np.int32)

    for b_idx, idxs in enumerate(block_idxs):
        preds_np = block_preds_list[b_idx]  # shape [M]
        class_counts[idxs, preds_np] += 1

    global_labels = class_counts.argmax(axis=1).astype(np.uint8)

    # 7. Compute Metrics
    print("Computing accuracy and metrics...")
    overall_accuracy, confusion, class_iou, mean_iou = compute_metrics(global_labels, ground_truth, args.num_classes)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Class IoU: {class_iou}")
    print(f"Confusion Matrix:\n{confusion}")

    # After computing `global_labels` and `ground_truth` in your main function:
    building_metrics = evaluate_building_accuracy(global_labels, ground_truth, building_label=1)
    print(f"Building-specific metrics: {building_metrics}")

    # 8. Write out new .las
    print(f"Writing output to {args.output_las}")
    if 'classification' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='classification', type='uint8'))
    las.classification = global_labels
    las.write(args.output_las)
    print("Done. Output saved to", args.output_las)

if __name__ == "__main__":
    main()
