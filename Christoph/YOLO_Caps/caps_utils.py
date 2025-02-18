import os 
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import hashlib

from PIL import Image
from collections import Counter
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data


class DICOMCoarseDataset(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, transform=None):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through all classes
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                # Find all JPEG files in the class folder
                jpeg_files = [f for f in os.listdir(class_folder) if f.endswith('.jpeg')]

                # Select images randomly if more images than needed
                if len(jpeg_files) > self.num_images_per_class:
                    selected_files = random.sample(jpeg_files, self.num_images_per_class)
                else:
                    selected_files = jpeg_files

                # Save paths and corresponding labels
                for file_name in selected_files:
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_label)
        
    # Return number of images
    def __len__(self):
        return len(self.image_paths) 
    
    # Return image and label
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        image = Image.fromarray(np.uint8(image))

        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    
    def get_labels(self):
        return self.labels
    
    #Plot label distribution
    def display_label_distribution(self):
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [self.classes[label] for label in labels])
        plt.show()
    
    #Visualize images
    def visualize_images(self, num_images=5):
        num_images = min(num_images, len(self.image_paths))
        _, axes = plt.subplots(1, num_images, figsize=(15,15))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            random_index = random.randint(0, len(self.image_paths)-1)
            image, label = self.__getitem__(random_index)
            if isinstance(image, torch.Tensor):
                image = image.squeeze().numpy()
            axes[i].imshow(image, cmap="gray")
            axes[i].set_title(f"Label: {self.classes[label]}")
            axes[i].axis("off")
        plt.show()

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        #Wraps an existing dataset and applies transformations
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        #Retrieves an item and applies the transformation if provided
        sample, label = self.base_dataset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        #Returns the size of the dataset
        return len(self.base_dataset)
 
def display_data_loader_batch(data_loader, classes):
    #Displays a batch of images from a DataLoader with their labels
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    num_images = min(len(images), 5)
    _, axes = plt.subplots(1, num_images, figsize=(15,15))
    if num_images == 1:
        axes = [axes]
    for i in range(num_images):
        image = images[i].cpu()
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() == 3:
            image = image.permute(1,2,0)
        image = image.numpy()
        # Normalize and adjust dimensions for display
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(axis=-1)  # Remove the channel dimension for grayscale
        elif image.ndim == 2:
            image = image  # Grayscale images should remain 2D
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    plt.show()



class CAPS_Productive_Dataset(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, transform=None):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate through classes
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                # Get all JPEG images in the class folder
                jpeg_files = [f for f in os.listdir(class_folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.JPEG')]

                # Select a subset of images if there are more than needed
                if len(jpeg_files) > self.num_images_per_class:
                    selected_files = random.sample(jpeg_files, self.num_images_per_class)
                # Use all images if fewer than required
                else:
                    selected_files = jpeg_files

                # Store file paths and their corresponding labels
                for file_name in selected_files:
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_label)
        

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        image = Image.fromarray(np.uint8(image))

        if self.transform:
            image = self.transform(image)
        # label = self.labels[index]
        label=os.path.basename(img_path)
        return image, label
    
    def get_labels(self):
        return self.labels
    
    def display_label_distribution(self):
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [self.classes[label] for label in labels])
        plt.show()
    
    def visualize_images(self, num_images=5):
        num_images = min(num_images, len(self.image_paths))
        _, axes = plt.subplots(1, num_images, figsize=(15,15))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            random_index = random.randint(0, len(self.image_paths)-1)
            image, label = self.__getitem__(random_index)
            if isinstance(image, torch.Tensor):
                image = image.squeeze().numpy()
            axes[i].imshow(image, cmap="gray")
            axes[i].set_title(f"Label: {self.classes[label]}")
            axes[i].axis("off")
        plt.show()