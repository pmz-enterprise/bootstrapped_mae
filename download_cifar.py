import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import yaml
from pathlib import Path

# Load the CIFAR10 training set
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

# Initialize variables
mean = torch.zeros(3)
std = torch.zeros(3)
total_images = 0

# Compute the mean
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    total_images += batch_samples
mean /= total_images

# Compute the standard deviation
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
std = torch.sqrt(std / total_images)

print(f'Mean: {mean.tolist()}')
print(f'Std: {std.tolist()}')