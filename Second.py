import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for a single image
def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load pre-trained ResNet model and modify for classification
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Ensure this matches the number of classes
model.load_state_dict(torch.load("model_epoch_1.pth", map_location=device), strict=False)  # Load trained weights)
model = model.to(device)
model.eval()

# Function to predict image class
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example usage
image_path = "/images.png"  # Change this to your image path
predicted_class = predict_image(image_path, model)
print(f"Predicted Class: {predicted_class}")
