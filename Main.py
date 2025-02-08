import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
def image_transform(image):
    return transforms.ToTensor()(image)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Lambda(image_transform),  # Use the custom transformation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets for 'one_molecule' category
data_dir = '/content/dataset_for_image_classifier/for_model'  # Change this path to your actual data path
train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = ImageFolder(root=f"{data_dir}/validation", transform=transform)
test_dataset = ImageFolder(root=f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load pre-trained ResNet model and modify for multi-class classification
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scaler = GradScaler()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {100 * correct/total:.2f}%")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth", _use_new_zipfile_serialization=False)
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {100 * acc:.2f}%")

# Evaluate the trained model
evaluate_model(model, test_loader)
