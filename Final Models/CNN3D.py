import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


class CNN3D(nn.Module):
    def __init__(self, image_size, classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(image_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, len(classes))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Custom3DTransform:
    def __init__(self, resize=(224, 224), num_channels=3, normalize_means=(0.485, 0.456, 0.406), normalize_stds=(0.229, 0.224, 0.225), flip_prob=0.5):
        self.resize_value = resize
        self.num_channels = num_channels
        self.normalize_means = normalize_means
        self.normalize_stds = normalize_stds
        self.flip_prob = flip_prob

    def grayscale(self, volume):
        # Convert each frame to grayscale and replicate channels if needed
        volume_gray = np.stack([np.array(Image.fromarray(frame).convert('L')) for frame in volume], axis=0)
        if self.num_channels == 3:  # Replicate grayscale across RGB channels if needed
            volume_gray = np.repeat(volume_gray[:, np.newaxis, ...], 3, axis=1)
        return volume_gray 
    
    def resize(self, volume):
        # Resize each frame
        return np.array([np.array(Image.fromarray(np.transpose(frame, (1,2,0))).resize(self.resize_value)) for frame in volume])
    
    def random_horizontal_flip(self, volume):
        if random.random() < self.flip_prob:
            return np.array([np.fliplr(frame) for frame in volume])
        return volume
    
    def normalize(self, volume):
        # Normalize each channel in each frame
        volume = (volume - np.array(self.normalize_means)[None, None, None, :]) / np.array(self.normalize_stds)[None, None, None, :]
        return volume
    
    def to_tensor(self, volume):
        return torch.tensor(volume, dtype=torch.float32).permute(1, 0, 2, 3)
    
    def __call__(self, volume):
        volume = self.grayscale(volume)
        volume = self.resize(volume)
        volume = self.random_horizontal_flip(volume)
        volume = self.normalize(volume)
        return self.to_tensor(volume)