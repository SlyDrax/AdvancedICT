#!/usr/bin/env python3

"""
preprocess.py
A script for converting and preprocessing a single LAZ file for PointNet segmentation.
- Converts .laz to .las
- Reads .las with laspy
- Maps classification codes to custom classes (terrain=0, building=1, road=2)
- Optionally normalizes and/or downsamples
- Splits into train/val sets
- Saves .npz for easy loading in train.py
"""

import os
import subprocess
import argparse

import laspy
import numpy as np

def convert_laz_to_las(input_laz, output_las):
    """
    Converts a .laz file to .las using laszip or LAStools.
    Requires laszip or las2las to be installed and in PATH.
    """
    # Example using laszip (other tools may require different commands)
    cmd = ["laszip", "-i", input_laz, "-o", output_las]
    print(f"Converting {input_laz} to {output_las} using laszip...")
    subprocess.run(cmd, check=True)
    print("Conversion complete.")

def load_las_file(las_path):
    """
    Loads a .las file using laspy.
    Returns x, y, z, intensity, classification.
    """
    print(f"Reading {las_path}...")
    las = laspy.read(las_path)

    # Extract point attributes
    x = las.x
    y = las.y
    z = las.z

    # Check if intensity is available
    intensity = getattr(las, 'intensity', None)
    if intensity is None:
        # If intensity doesn't exist, create a dummy array of zeros or ones
        intensity = np.zeros_like(x)

    # Classification (LAS standard or custom)
    labels = las.classification

    return x, y, z, intensity, labels

def map_labels_to_custom_classes(labels, class_map):
    """
    Map the raw LAS classification codes to custom classes.
    class_map is a dictionary: {las_class_code: custom_label}
    E.g. {2:0 (terrain), 6:1 (building), 11:2 (road)}
    Everything not in class_map is mapped to a default class, e.g. 0.
    """
    mapped = np.array([class_map.get(l, 0) for l in labels], dtype=np.int64)
    return mapped

def normalize_point_cloud(x, y, z):
    """
    Centers and (optionally) scales the point cloud.
    Commonly done to improve neural network training stability.
    """
    # Center the point cloud
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    x -= mean_x
    y -= mean_y
    z -= mean_z

    # Optionally scale (comment out if not desired)
    scale = np.max(np.sqrt(x**2 + y**2 + z**2))
    if scale > 0:
        x /= scale
        y /= scale
        z /= scale
    
    return x, y, z

def subsample_point_cloud(points, labels, max_points=1000000):
    """
    Randomly subsample the point cloud to a maximum of `max_points`.
    If the point cloud is already smaller, returns unchanged arrays.
    """
    N = points.shape[0]
    if N > max_points:
        print(f"Subsampling from {N} to {max_points} points...")
        idx = np.random.choice(N, max_points, replace=False)
        points = points[idx]
        labels = labels[idx]
    return points, labels

def split_train_val(points, labels, train_ratio=0.8):
    """
    Splits the dataset into train and val sets by random sampling.
    """
    N = points.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_count = int(train_ratio * N)
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_points = points[train_idx]
    train_labels = labels[train_idx]
    val_points = points[val_idx]
    val_labels = labels[val_idx]

    return train_points, train_labels, val_points, val_labels

def main():
    parser = argparse.ArgumentParser(description="Preprocess .laz data for PointNet segmentation.")
    parser.add_argument("--input_laz", type=str, required=True, help="Path to input .laz file.")
    parser.add_argument("--output_npz", type=str, default="dataset.npz", help="Output .npz file.")
    parser.add_argument("--max_points", type=int, default=1000000, help="Max points to subsample.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio.")
    args = parser.parse_args()

    # 1. Convert .laz -> .las
    base_name = os.path.splitext(os.path.basename(args.input_laz))[0]
    temp_las_path = f"{base_name}.las"
    convert_laz_to_las(args.input_laz, temp_las_path)

    # 2. Load .las
    x, y, z, intensity, las_labels = load_las_file(temp_las_path)

    # 3. Map LAS classifications to custom classes
    # Example class_map: {2:0, 6:1, 11:2}, everything else -> 0
    class_map = {
        2: 0,   # Terrain
        6: 1,   # Building
        11: 2   # Road (example code, adjust if needed)
    }
    mapped_labels = map_labels_to_custom_classes(las_labels, class_map)

    # 4. Normalize point cloud (optional but recommended)
    x, y, z = normalize_point_cloud(x, y, z)

    # 5. Combine features (XYZ + intensity)
    points = np.stack([x, y, z, intensity], axis=-1)  # [N, 4]

    # 6. (Optional) Subsample the point cloud
    points, mapped_labels = subsample_point_cloud(points, mapped_labels, max_points=args.max_points)

    # 7. Split into train/val sets
    train_points, train_labels, val_points, val_labels = split_train_val(
        points, mapped_labels, train_ratio=args.train_ratio
    )

    # 8. Save to .npz
    print(f"Saving dataset to {args.output_npz}...")
    np.savez(
        args.output_npz,
        train_points=train_points,
        train_labels=train_labels,
        val_points=val_points,
        val_labels=val_labels
    )
    print("Done.")

    # Clean up: remove temporary .las file if desired
    # (Uncomment to remove the .las file automatically)
    # os.remove(temp_las_path)

if __name__ == "__main__":
    main()
