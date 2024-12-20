import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet.model import PointNetDenseCls 
import random
from torch.cuda.amp import GradScaler, autocast
import os


class BlockDataset(Dataset):
    def __init__(self, block_points, block_labels, num_points=2048, augment=False):
        self.block_points = block_points
        self.block_labels = block_labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.block_points)

    def __getitem__(self, idx):
        points = self.block_points[idx]  
        labels = self.block_labels[idx] 
        M = points.shape[0]

        if M > self.num_points:
            
            idxs = np.random.choice(M, self.num_points, replace=False)
            points = points[idxs]
            labels = labels[idxs]
        elif M < self.num_points:
           
            shortfall = self.num_points - M
            replicate_idxs = np.random.choice(M, shortfall, replace=True)
            points = np.concatenate([points, points[replicate_idxs]], axis=0)
            labels = np.concatenate([labels, labels[replicate_idxs]], axis=0)

        if self.augment:
            theta = random.uniform(-np.pi, np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            points[:, :3] = points[:, :3].dot(rotation_matrix)

        points = torch.from_numpy(points).float() 
        labels = torch.from_numpy(labels).long()   
        return points, labels


def load_npz_files_from_folder(folder_path):
    all_block_points = []
    all_block_labels = []

    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} .npz files in folder: {folder_path}")

    for npz_file in npz_files:
        file_path = os.path.join(folder_path, npz_file)
        print(f"Loading {file_path}...")
        data = np.load(file_path, allow_pickle=True)

        all_block_points.extend(data['block_points'])
        all_block_labels.extend(data['block_labels'])

    print(f"Loaded total blocks: {len(all_block_points)}")
    return np.array(all_block_points, dtype=object), np.array(all_block_labels, dtype=object)


def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes, scaler):
    model.train()
    running_loss = 0.0
    for batch_data in dataloader:
        points, labels = batch_data 
        points, labels = points.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if points.shape[-1] == 4:
            points = points[..., :3] 

        points = points.permute(0, 2, 1)

        optimizer.zero_grad()
        with autocast():
            seg_logits, _, _ = model(points)
            seg_logits = seg_logits.view(-1, num_classes)
            labels = labels.view(-1)

            loss = criterion(seg_logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="preprocessed", help="Folder containing .npz files.")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample per block")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_out", type=str, default="pointnet_block.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    block_points, block_labels = load_npz_files_from_folder(args.input_folder)

    train_dataset = BlockDataset(block_points, block_labels, num_points=args.num_points)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True
    )

    # Initialize the model, loss, optimizer, and scaler
    model = PointNetDenseCls(k=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Training loop
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.num_classes, scaler)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{args.model_out}_epochs={epoch + 1}.pth")
            print(f"Model saved to {args.model_out}_epochs={epoch + 1}.pth")

    print("Model finished training")


if __name__ == "__main__":
    main()
