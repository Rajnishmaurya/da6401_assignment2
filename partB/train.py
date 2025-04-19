#Fine-Tuning Last k Layers with ResNet50

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import wandb
import os

# Initialize Weights & Biases
wandb.init(project="assignment2_partB", config={
    "model": "resnet50",
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "k": 20  # Number of layers to unfreeze
})
config = wandb.config

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
train_path = '/kaggle/input/nature/inaturalist_12K/train'
val_path = '/kaggle/input/nature/inaturalist_12K/val'

# Full train set
full_train_dataset = ImageFolder(train_path, transform=transform)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Test set from the original validation folder
test_dataset = ImageFolder(val_path, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # Replace final layer for 10 classes
model = model.to(device)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last k layers
ct = 0
for child in list(model.children())[::-1]:
    ct += 1
    if ct <= config.k:
        for param in child.parameters():
            param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

# Training loop
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    val_acc = 0
    val_loss = 0


    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        val_acc = 100. * correct / total

    wandb.log({
        "epoch": epoch+1,
        "train_loss": running_loss / len(train_loader),
        "train_acc": train_acc,
        "val_loss": val_loss / len(val_loader),
        "val_acc": val_acc
    })

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}%")

# Final test evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
wandb.log({"final_test_accuracy": test_accuracy})
