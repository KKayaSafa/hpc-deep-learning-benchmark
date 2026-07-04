import torchvision
import warnings
import os
import torch
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights # Changed to EfficientNetB2

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

# Download pretrained EfficientNetB2 weights
print("Downloading pretrained EfficientNetB2 weights...")
weights = EfficientNet_B2_Weights.DEFAULT # Changed to EfficientNet_B2_Weights
model = efficientnet_b2(weights=weights) # Changed to efficientnet_b2

# Save weights locally
weights_path = './model-weights/efficientnetb2_weights.pth' # Changed path name
torch.save(model.state_dict(), weights_path)

print(f"EfficientNetB2 pretrained weights saved successfully at {weights_path}")
print(f"Model architecture: EfficientNetB2 with compound scaling")
print(f"Input resolution: 288x288")
print(f"Parameters: ~9.2M")
print(f"FLOPs: ~1.0B")