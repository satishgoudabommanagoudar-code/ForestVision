import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------
# Settings
# -----------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "model/forest_model.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Load Dataset
# -----------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------
# Model
# -----------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------
# Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------
# Training
# -----------------------
print("Training Started...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss:.4f}")

print("\nTraining Completed!")

os.makedirs("model", exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")