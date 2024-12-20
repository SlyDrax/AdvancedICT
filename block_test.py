import argparse
import laspy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pointnet.model import PointNetDenseCls
from tqdm import tqdm  # For progress bars

def create_blocks_inference(points, block_size=50.0, stride=25.0):
    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])

    blocks = []
    block_idxs = []

    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            x_cond = (points[:,0] >= x) & (points[:,0] < x + block_size)
            y_cond = (points[:,1] >= y) & (points[:,1] < y + block_size)
            cond = x_cond & y_cond

            sub_points = points[cond]
            idxs = np.where(cond)[0]  

            if sub_points.shape[0] > 0:
                # local normalization
                centroid = np.mean(sub_points[:,:3], axis=0)
                sub_points[:,:3] -= centroid
                dist = np.max(np.linalg.norm(sub_points[:,:3], axis=1))
                if dist > 0:
                    sub_points[:,:3] /= dist

                blocks.append(sub_points)
                block_idxs.append(idxs)

            y += stride
        x += stride

    return blocks, block_idxs

class InferenceBlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block_points = self.blocks[idx] 
        return block_points

def run_inference(model, dataset, batch_size, num_classes, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    block_preds_list = []
    with torch.no_grad():
        for block_batch in tqdm(loader, desc="Processing Blocks"):
            block_batch = block_batch.to(device).float()

            if block_batch.shape[-1] == 4:
                block_batch = block_batch[:,:,:3]

            block_batch = block_batch.permute(0, 2, 1)

            seg_logits, _, _ = model(block_batch)
            pred_classes = seg_logits.argmax(dim=2)

            block_preds_list.extend(pred_classes.cpu().numpy()) 

    return block_preds_list

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Block Inference")
    parser.add_argument("--input_las", type=str, required=True, help="Path to new .las file for inference")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Trained model .pth")
    parser.add_argument("--output_las", type=str, default="inferred.las", help="Output .las with predicted classes")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--block_size", type=float, default=50.0, help="Size of each block in meters")
    parser.add_argument("--stride", type=float, default=25.0, help="Stride for block sampling in meters")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Reading LAS file: {args.input_las}")
    las = laspy.read(args.input_las)
    x, y, z = las.x, las.y, las.z
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    print("Intensity data not used.")

    print("Partitioning point cloud into blocks...")
    blocks, block_idxs = create_blocks_inference(points,
                                                 block_size=args.block_size,
                                                 stride=args.stride)

    if len(blocks) == 0:
        print("No blocks found. Check your block_size/stride or .las content.")
        return

    print(f"Total blocks created: {len(blocks)}")

    inference_dataset = InferenceBlockDataset(blocks)

    print(f"Loading model from: {args.model_ckpt}")
    model = PointNetDenseCls(k=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    print("Running inference on blocks...")
    block_preds_list = run_inference(model, inference_dataset,
                                     1, args.num_classes, device)
    print("Inference completed.")

    print("Stitching predictions back to the global point cloud using majority voting...")
    num_points = points.shape[0]
    num_classes = args.num_classes
    class_counts = np.zeros((num_points, num_classes), dtype=np.int32)

    for b_idx, idxs in enumerate(block_idxs):
        preds_np = block_preds_list[b_idx]
        class_counts[idxs, preds_np] += 1

    global_labels = class_counts.

    print(f"Writing output to {args.output_las}")

    if 'classification' not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='classification', type='uint8'))
    las.classification = global_labels
    las.write(args.output_las)
    print("Done. Output saved to", args.output_las)

if __name__ == "__main__":
    main()
