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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_transform(image):
    return transforms.ToTensor()(image)

train_val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Lambda(image_transform),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For inference (no augmentation)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = '/content/dataset_for_image_classifier/for_model'  # Replace with your path
train_dataset = ImageFolder(f"{data_dir}/train", transform=train_val_test_transform)
val_dataset = ImageFolder(f"{data_dir}/validation", transform=train_val_test_transform)
test_dataset = ImageFolder(f"{data_dir}/test", transform=train_val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scaler = GradScaler()

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

        # Validation
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

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

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

def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# If you’ve already trained the model:
model.load_state_dict(torch.load("model_epoch_1.pth", map_location=device), strict=False)
model.eval()

# Evaluate (optional)
evaluate_model(model, test_loader)

# Predict (optional)
image_path = "/images.png"  # Replace with your image
predicted_class = predict_image(image_path, model)
print(f"Predicted Class: {predicted_class}")
