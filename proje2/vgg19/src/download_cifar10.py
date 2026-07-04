import torchvision
import warnings
import os
import torch
from torchvision.models import vgg19, VGG19_Weights # Changed to VGG19

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

# Download pretrained VGG19 weights
print("Downloading pretrained VGG19 weights...")
weights = VGG19_Weights.DEFAULT # Changed to VGG19_Weights
model = vgg19(weights=weights) # Changed to vgg19

# Save weights locally
weights_path = './model-weights/vgg19_weights.pth' # Changed path name
torch.save(model.state_dict(), weights_path)

print(f"VGG19 pretrained weights saved successfully at {weights_path}")
print(f"Model architecture: VGG19 with batch normalization")