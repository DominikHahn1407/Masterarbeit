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
    def __init__(self, root_dir, num_images_per_class, classes, transform=None, scenario=1):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Durchlaufe die Klassen (z. B. 'nodule' und 'non-nodule')
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                # Finde alle JPEG-Bilder
                jpeg_files = [f for f in os.listdir(class_folder) if f.endswith('.jpeg')]

                # Wähle zufällig aus, wenn mehr Bilder als benötigt vorhanden sind
                if len(jpeg_files) > self.num_images_per_class:
                    selected_files = random.sample(jpeg_files, self.num_images_per_class)
                # Wähle alle Bilder aus, wenn weniger Bilder als benötigt vorhanden sind
                else:
                    selected_files = jpeg_files

                # Speichere die Pfade und zugehörigen Labels
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

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        sample, label = self.base_dataset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.base_dataset)
 
def display_data_loader_batch(data_loader, classes):
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


    def __init__(self, root_dir, classes, transform=None, scenario=1, balance_n=True):
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in os.listdir(root_dir):
            for file_name in os.listdir(os.path.join(root_dir, folder)):
                if file_name.endswith(".dcm"):
                    prefix = file_name[0]
                    if folder == "non-nodule":
                        if scenario == 2 and prefix != "N":
                            continue
                        if scenario == 3 and prefix == "N":
                            continue
                        prefix = "N"
                    if prefix in self.classes:
                        file_name = os.path.join(root_dir, folder, file_name)
                        self.image_paths.append(file_name)
                        self.labels.append(self.classes[prefix])
        if balance_n:
            self._balance_class_n()

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
    
    def _balance_class_n(self):
        label_counts = Counter(self.labels)
        counts = sorted(label_counts.values(), reverse=True)
        max_amount = counts[1]
        n_indices = [i for i, label in enumerate(self.labels) if label == self.classes["N"]]
        if len(n_indices) > max_amount:
            n_indices = random.sample(n_indices, max_amount)
        balanced_indices = [
            i for i, label in enumerate(self.labels) if label != self.classes["N"]
        ] + n_indices
        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]

    def get_labels(self):
        return self.labels
    
    def display_label_distribution(self):
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [list(self.classes.keys())[label] for label in labels])
        plt.show()

    def visualize_images(self, num_images=5):
        num_images = min(num_images, len(self.image_paths))
        _, axes = plt.subplots(1, num_images, figsize=(15, 15))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            random_index = random.randint(0, len(self.image_paths) - 1)
            image, label = self.__getitem__(random_index)
            if isinstance(image, torch.Tensor):
                image = image.squeeze().numpy()
            axes[i].imshow(image, cmap="gray")
            axes[i].set_title(f"Label: {list(self.classes.keys())[label]}")
            axes[i].axis("off")
        plt.show()


class CAPS_Productive_Dataset(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, transform=None, scenario=1):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Durchlaufe die Klassen (z. B. 'nodule' und 'non-nodule')
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                # Finde alle JPEG-Bilder
                jpeg_files = [f for f in os.listdir(class_folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.JPEG')]

                # Wähle zufällig aus, wenn mehr Bilder als benötigt vorhanden sind
                if len(jpeg_files) > self.num_images_per_class:
                    selected_files = random.sample(jpeg_files, self.num_images_per_class)
                # Wähle alle Bilder aus, wenn weniger Bilder als benötigt vorhanden sind
                else:
                    selected_files = jpeg_files

                # Speichere die Pfade und zugehörigen Labels
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