import torchvision
import warnings
import os
import torch
from torchvision.models import inception_v3, Inception_V3_Weights # Changed from ResNet50

# Ignore warning messages
warnings.filterwarnings("ignore")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Download the training dataset
print("Downloading CIFAR10 training dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=None
)

# Download the test dataset
print("Downloading CIFAR10 test dataset...")
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=None
)

print(f"CIFAR10 dataset downloaded successfully!")
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create directory to store weights
os.makedirs('./model-weights', exist_ok=True)

# Download pretrained InceptionV3 weights
print("Downloading pretrained InceptionV3 weights...")
weights = Inception_V3_Weights.DEFAULT # Changed to Inception_V3_Weights
model = inception_v3(weights=weights) # Changed to inception_v3

# Save weights locally
weights_path = './model-weights/inceptionv3_weights.pth' # Changed path name
torch.save(model.state_dict(), weights_path)

print(f"InceptionV3 pretrained weights saved successfully at {weights_path}")