import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import AlzheimerCNN

DATA_PATH = r"K:\alzheimer_project\Dataset"

BATCH_SIZE = 16
EPOCHS = 8
LR = 0.001
IMG_SIZE = 128

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

num_classes = len(dataset.classes)
print("Detected classes:", dataset.classes)

train_size = int(0.6 * len(dataset))
val_size   = int(0.2 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: pass number of classes dynamically
model = AlzheimerCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Total images: {len(dataset)}")
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {test_size}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# ---- Validation ----
correct, total = 0, 0
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total * 100
print(f"Validation Accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), "alzheimer_model.pth")
print("Model saved as alzheimer_model.pth")
