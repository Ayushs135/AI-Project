# 🧪 AI_Project — Chemical Image Classification with ResNet-18

This repository contains a complete deep learning pipeline to classify chemical structure images using a fine-tuned [ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) model in PyTorch.

The model is trained to recognize **4 different categories of chemical images**, using transfer learning and modern training techniques such as **automatic mixed precision (AMP)** for improved speed and efficiency.

---

## 📊 Dataset

The dataset used in this project can be found on Kaggle:

> 🔗 [Images Dataset for Chemical Images Classifier](https://www.kaggle.com/datasets/alekseikrasnov/images-dataset-for-chemical-images-classifier)

The dataset is organized into `train`, `validation`, and `test` folders, each containing subfolders for the 4 image classes.

---

## 🚀 Getting Started in Google Colab

You can use [Google Colab](https://colab.research.google.com/) for easy access to GPU resources and dataset integration via Kaggle.

### ✅ Step-by-step:

```python
# Install Kaggle API
!pip install kaggle

# Upload kaggle.json (your API key)
from google.colab import files
files.upload()

# Set up kaggle.json path
import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
!chmod 600 /root/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d alekseikrasnov/images-dataset-for-chemical-images-classifier

# Unzip the dataset
!unzip /content/images-dataset-for-chemical-images-classifier.zip
```

---

## 🧠 Model Architecture

- **Base Model**: Pretrained ResNet-18 from `torchvision.models`
- **Modified Output Layer**: `nn.Linear(..., 4)` for 4-class classification
- **Optimizer**: Adam (`lr = 0.0005`)
- **Loss Function**: Cross Entropy Loss
- **Training Enhancements**:
  - Mixed Precision with `torch.amp`
  - Data augmentation (resize, rotation, flip)
  - Evaluation on validation and test sets

---

## 🏋️ Training

The script trains the model and saves the checkpoint after each epoch:

```python
# To train the model (in script)
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)
```

Model checkpoints are saved as:
```bash
model_epoch_1.pth
```

---

## 🧪 Evaluation

After training, the model is evaluated on the test set with accuracy reported using `sklearn.metrics.accuracy_score`.

---

## 🔍 Inference: Predict a Single Image

You can use the script to load a trained model and predict the class of a new image:

```python
image_path = "/path/to/image.png"
predicted_class = predict_image(image_path, model)
print(f"Predicted Class: {predicted_class}")
```

Ensure the image is preprocessed similarly (resized, normalized).

---

## 🧾 Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- PIL
- Google Colab (optional)

---

## ✨ Acknowledgements

- Dataset by [Aleksei Krasnov](https://www.kaggle.com/datasets/alekseikrasnov)
- Pretrained ResNet-18 by PyTorch
- Mixed precision training via `torch.amp`

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out!
