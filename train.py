import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LidarDataset
from pointnet.model import PointNetDenseCls

DATA_PATH = 'dataset.npz'
NUM_POINTS = 2048
BATCH_SIZE = 16
LR = 0.001
NUM_EPOCHS = 50
NUM_CLASSES = 3  # building, road, terrain

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for points, labels in dataloader:
        # points: [B, N, 3] or [B, N, 4]
        # labels: [B, N]
        points, labels = points.to(device), labels.to(device)

        # If points is [B, N, 4], slice off intensity:
        points = points[:, :, :3]
        
        # Now permute to [B, 3, N]
        points = points.permute(0, 2, 1)
        
        optimizer.zero_grad()
        seg_logits, trans, trans_feat = model(points)  # seg_logits: [B, N, NUM_CLASSES]
        
        # For cross-entropy, reshape to 2D
        # seg_logits: [B, N, 3] -> [B*N, 3]
        seg_logits_2d = seg_logits.view(-1, NUM_CLASSES)
        labels_1d = labels.view(-1)  # shape [B*N]
        
        loss = criterion(seg_logits_2d, labels_1d)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LidarDataset(DATA_PATH, split='train', num_points=NUM_POINTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = PointNetDenseCls(k=NUM_CLASSES, feature_transform=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "pointnet_segmentation.pth")

if __name__ == '__main__':
    main()
