import os 
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from PIL import Image
from collections import Counter
from torch.utils.data import Dataset

class DICOMCoarseDataset(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, transform=None):
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []


        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                dicom_files = [f for f in os.listdir(class_folder) if f.endswith('.dcm')]
                selected_files = dicom_files[:self.num_images_per_class]
                for file_name in selected_files:
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
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