import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import rotate, resized_crop, affine


class CNN3D(nn.Module):
    def __init__(self, image_size, classes):
        super(CNN3D, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        # Calculate the flattened size after convolutions
        # Assuming input size is (32, 1, 16, 224, 224) and two conv + maxpool layers
        # After the first Conv3D shape will be (32, 32, 14, 222, 222)
        # After the first MaxPool3D, shape will be (32, 32, 7, 111, 111)
        # After the second Conv3D, shape will be (32, 64, 5, 109, 109)
        # After the second MaxPool3D, shape will be (32, 64, 2, 54, 54)
        self.fc1 = nn.Linear(64 * 2 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, len(classes))
        self.relu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.15)
        
    def _conv_layer_set(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2)),
        )
    
    def forward(self, x):
        # Input shape: [batch_size, 1, num_slices, image_size, image_size]
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)

        # Flatten output for fc
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Custom3DTransform:
    def __init__(self, resize=(224,224), data_augmentation=False, num_channels=1, normalize_means=0.485, normalize_stds=0.229, flip_prob=0.5):
        super(Custom3DTransform, self).__init__()
        self.resize_value = resize
        self.data_augmentation = data_augmentation
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
        return np.array([np.array(Image.fromarray(frame).resize(self.resize_value)) for frame in volume])
    
    def random_horizontal_flip(self, volume):
        if random.random() < self.flip_prob:
            return np.array([np.fliplr(frame) for frame in volume])
        return volume
    
    def random_rotation(self, volume, degrees=15):
        angle = random.uniform(-degrees, degrees)
        return np.array([np.array(rotate(Image.fromarray(frame), angle)) for frame in volume])
    
    def random_resized_crop(self, volume, scale=(0.8, 1.2), ratio=(0.9, 1.1)):
        h, w = volume.shape[1:3]
        crop_h = min(int(h * random.uniform(*scale)),h)
        crop_w = min(int(w * random.uniform(*ratio)),w)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        return np.array([
            np.array(resized_crop(Image.fromarray(frame), top, left, crop_h, crop_w, self.resize_value))
            for frame in volume
        ])

    def random_shear(self, volume, shear=10):
        shear_x = random.uniform(-shear, shear)
        shear_y = random.uniform(-shear, shear)
        return np.array([
            np.array(affine(Image.fromarray(frame), angle=0, translate=(0,0), scale=1.0, shear=(shear_x, shear_y))) 
            for frame in volume
        ])

    def normalize(self, volume):
        # Normalize each channel in each frame
        volume = (volume - self.normalize_means) / self.normalize_stds
        return volume
    
    def to_tensor(self, volume):
        volume = np.expand_dims(volume, axis=0)
        return torch.tensor(volume, dtype=torch.float32)
    
    def __call__(self, volume):#need to be overwritten cause the dataloader does overwrite init function
        volume = self.grayscale(volume)
        if self.data_augmentation:
            volume = self.random_horizontal_flip(volume)
            volume = self.random_rotation(volume)
            volume = self.random_resized_crop(volume)
            volume = self.random_shear(volume)
        else:
            volume = self.resize(volume)
        volume = self.normalize(volume)
        return self.to_tensor(volume)