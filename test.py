#!/usr/bin/env python3

"""
test.py
Runs inference (point-wise classification) on a .las file using a trained PointNetDenseCls model,
and writes out a new .las file with predicted labels in the classification field.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import laspy
from torch.utils.data import DataLoader, Dataset
from pointnet.model import PointNetDenseCls  # or wherever your model is located

class LASDataset(Dataset):
    """
    A simple PyTorch Dataset for loading all points from a .las file.
    We only do inference here, so no labels needed.
    """
    def __init__(self, las_path, use_intensity=False):
        super().__init__()
        self.las_path = las_path
        self.use_intensity = use_intensity
        print(f"Reading LAS file: {las_path}")
        las = laspy.read(las_path)

        self.x = las.x
        self.y = las.y
        self.z = las.z
        
        if hasattr(las, 'intensity') and use_intensity:
            self.intensity = las.intensity
            # shape [N, 4]
            self.points = np.stack([self.x, self.y, self.z, self.intensity], axis=-1)
        else:
            # shape [N, 3]
            self.points = np.stack([self.x, self.y, self.z], axis=-1)

        self.points = self.points.astype(np.float32)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        # Returns a single point [3] or [4]
        return self.points[idx]

def inference(model, las_dataset, batch_size=8192, num_classes=3, use_intensity=False, device='cpu'):
    """
    Runs model inference over an entire LAS dataset in batches.
    Returns an array of predicted labels, shape [N].
    """
    model.eval()
    dataloader = DataLoader(las_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_preds = []
    with torch.no_grad():
        for batch_points in dataloader:
            # batch_points: [B, 3 or 4]
            batch_points = batch_points.to(device)
            # Reshape for model: [B, C, 1] if we treat each point separately
            # But PointNetDenseCls expects [B, 3, N] for an entire chunk of points.
            # We'll treat each batch as a "mini point-cloud" of size N=1 for each point 
            # which isn't typical usage, but let's do a trick:
            # Trick approach: We want to keep dims consistent with [B, C, N].
            
            # Another approach is to process entire dataset at once, but that might cause memory issues.
            # Let's handle it point by point in mini-batches:
            if use_intensity:
                # shape [B, 4] => we only have a 4D input, 
                # but the PointNet is coded for 3D. If your model doesn't handle intensity, slice off intensity:
                batch_points = batch_points[:, :3]
            
            # Now [B, 3], unsqueeze dimension to become [B, 3, 1] 
            # so the model sees "1 point per batch item"
            batch_points = batch_points.unsqueeze(-1)  # shape: [B, 3, 1]
            
            seg_logits, _, _ = model(batch_points)  # seg_logits: [B, 1, k]
            # shape is [B, 1, num_classes], we can squeeze the middle dim
            # Actually, the model returns [B, N, k], but we have N=1 here, so shape is [B, 1, k].
            seg_logits = seg_logits.squeeze(1)  # [B, k]
            
            # Class probabilities
            pred_classes = seg_logits.argmax(dim=1).cpu().numpy()  # shape [B]
            all_preds.append(pred_classes)

    all_preds = np.concatenate(all_preds, axis=0)  # shape [N]
    return all_preds

def main():
    parser = argparse.ArgumentParser(description="Run PointNet classification/segmentation on a .las file.")
    parser.add_argument("--input_las", type=str, required=True, help="Path to input LAS file")
    parser.add_argument("--output_las", type=str, default="classified_output.las", help="Path to output LAS file")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to the trained model .pth file")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (e.g. 3 for building/road/terrain)")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for inference")
    parser.add_argument("--use_intensity", action="store_true", help="If set, attempts to use intensity channel.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the dataset
    las_dataset = LASDataset(args.input_las, use_intensity=args.use_intensity)

    # 2. Load the model
    # We assume a segmentation-style model that outputs per-point labels (like PointNetDenseCls).
    model = PointNetDenseCls(k=args.num_classes, feature_transform=False).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    # 3. Run inference
    predicted_labels = inference(
        model, las_dataset, 
        batch_size=args.batch_size, 
        num_classes=args.num_classes, 
        use_intensity=args.use_intensity,
        device=device
    )

    # 4. Write out new LAS with predicted classes
    print(f"Writing predictions to {args.output_las}...")
    las = laspy.read(args.input_las)
    # If the las doesn't have classification dimension, add it
    # If it does, we'll overwrite the classification field
    if not hasattr(las, 'Classification'):
        # Create a new classification dimension if needed
        las.add_extra_dim(laspy.ExtraBytesParams(name="Classification", type="uint8"))
    # predicted_labels is shape [N], same length as las.x
    las.classification = predicted_labels.astype(np.uint8)
    las.write(args.output_las)
    print(f"Done. Output saved to {args.output_las}")

if __name__ == "__main__":
    main()
