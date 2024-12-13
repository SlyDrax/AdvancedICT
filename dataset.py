import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LidarDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed LiDAR data (.npz) for segmentation.
    Expects .npz to contain:
        - train_points, train_labels
        - val_points, val_labels
    Each of train_points or val_points is typically shaped [N, C], e.g. (x,y,z,intensity)
    Each of train_labels or val_labels is shaped [N].
    Args:
        npz_path: Path to the .npz file.
        split:    'train' or 'val'.
        num_points: Number of points to sample per item (for batch training).
        augment:    Whether to apply data augmentations (random jitter, rotations, etc).
    """
    def __init__(self, npz_path, split='train', num_points=2048, augment=False):
        super().__init__()
        self.npz_path = npz_path
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # Load data from .npz
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"File {self.npz_path} not found.")
        
        data = np.load(self.npz_path)
        
        if split == 'train':
            self.points = data['train_points']  # shape: [N, C]
            self.labels = data['train_labels']  # shape: [N]
        elif split == 'val':
            self.points = data['val_points']    # shape: [N, C]
            self.labels = data['val_labels']    # shape: [N]
        else:
            raise ValueError("split must be either 'train' or 'val'")

        # Convert to float32 / long for PyTorch
        self.points = self.points.astype(np.float32)
        self.labels = self.labels.astype(np.int64)

        # The entire set (N points) is treated as a pool we sample from 
        # in each __getitem__ call. This approach works well for "infinite" sampling.
        # Alternatively, you could store small blocks or patches as separate items.

    def __len__(self):
        # If you treat the entire point cloud as one big set, 
        # an "epoch" size could be arbitrary. 
        # One approach is to define the dataset length as the number 
        # of samples you'd like to draw. For example:
        return 10000  # You can set any "virtual" length if you're sampling randomly.

    def __getitem__(self, idx):
        """
        Returns a random subset of points (num_points) from the entire dataset, 
        along with the corresponding labels.
        """
        N = self.points.shape[0]

        # Randomly sample points for this item
        chosen_indices = np.random.choice(N, self.num_points, replace=False)
        sampled_points = self.points[chosen_indices]  # shape: [num_points, C]
        sampled_labels = self.labels[chosen_indices]  # shape: [num_points]

        if self.augment:
            sampled_points = self.data_augmentation(sampled_points)

        # Convert to PyTorch tensors
        sampled_points = torch.from_numpy(sampled_points)  # [num_points, C]
        sampled_labels = torch.from_numpy(sampled_labels)  # [num_points]

        return sampled_points, sampled_labels

    def data_augmentation(self, points):
        """
        Optionally apply random jitter, rotation, etc. for data augmentation.
        This helps the model generalize better.
        """
        # Example: small random jitter
        jitter = np.random.normal(0, 0.01, size=points.shape)
        points[:, :3] += jitter[:, :3]  # only jitter XYZ, not intensity
        # Additional augmentations can be added here (e.g., random rotation around Z)
        return points


def _test_lidar_dataset():
    """
    Quick test routine for the LidarDataset class.
    """
    dataset = LidarDataset('dataset.npz', split='train', num_points=2048, augment=True)
    print(f"Dataset length (virtual): {len(dataset)}")
    points, labels = dataset[0]  # get item
    print("Sampled points shape:", points.shape)
    print("Sampled labels shape:", labels.shape)
    print("Points dtype:", points.dtype, "Labels dtype:", labels.dtype)

if __name__ == "__main__":
    _test_lidar_dataset()
