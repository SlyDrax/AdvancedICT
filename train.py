import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LidarDataset
from pointnet.model import PointNetDenseCls  # segmentation model for example

# Config
DATA_PATH = 'dataset.npz'
NUM_POINTS = 2048
BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 4
NUM_CLASSES = 3

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_data in dataloader:
        points, labels = batch_data
        points, labels = points.to(device), labels.to(device)

        # If points is shape [B, N, 4], slice or adapt if the model expects [B, 3, N]
        points = points[..., :3]   # keep XYZ only if model doesn't handle intensity
        points = points.permute(0, 2, 1)  # [B, 3, N]

        optimizer.zero_grad()
        seg_logits, _, _ = model(points)  # seg_logits: [B, N, NUM_CLASSES]
        seg_logits = seg_logits.view(-1, NUM_CLASSES)
        labels = labels.view(-1)
        loss = criterion(seg_logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LidarDataset(DATA_PATH, split='train', num_points=NUM_POINTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=16, pin_memory=True)

    model = PointNetDenseCls(k=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}")
    # Save the final model
    torch.save(model.state_dict(), "pointnet_segmentation.pth")

if __name__ == '__main__':
    main()
