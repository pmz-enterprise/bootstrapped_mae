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
print('Computing mean...')
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    total_images += batch_samples
mean /= total_images

# Compute the standard deviation
print('Computing std...')
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
std = torch.sqrt(std / total_images)

# Prepare data
data = {
    'dataset': 'CIFAR10',
    'normalization': {
        'mean': mean.tolist(),
        'std': std.tolist()
    },
    'total_images': total_images
}

# Save to YAML file
output_path = Path('./configs/cifar10_normalization.yaml')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print(f'Results saved to {output_path}')
print(f'Mean: {mean.tolist()}')
print(f'Std: {std.tolist()}')