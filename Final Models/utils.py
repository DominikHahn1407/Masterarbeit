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
    def __init__(self, root_dir, num_images_per_class, classes, transform=None, scenario=1):
        random.seed(41)
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

                if class_name == "non-nodule":
                    if scenario == 2:
                        dicom_files = [f for f in dicom_files if f.startswith('N')]
                    elif scenario == 3:
                        dicom_files = [f for f in dicom_files if not f.startswith('N')]
                if len(dicom_files) >= self.num_images_per_class:
                    selected_files = random.sample(dicom_files, self.num_images_per_class)
                else:
                    selected_files = dicom_files
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


class DICOMFineDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for file_name in os.listdir(root_dir):
            if file_name.endswith(".dcm"):
                prefix = file_name[0]
                if prefix in self.classes:
                    self.image_paths.append(os.path.join(root_dir, file_name))
                    self.labels.append(self.classes[prefix])

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


class DicomCoarseDataset3D(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, transform=None, scenario=1, num_slices=16):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.num_slices = num_slices
        self.image_volumes = []
        self.labels = []

        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                dicom_files = sorted([f for f in os.listdir(class_folder) if f.endswith('.dcm')])

                # Scenario handling for non-nodule files
                if class_name == "non-nodule":
                    if scenario == 2:
                        dicom_files = [f for f in dicom_files if f.startswith('N')]
                    elif scenario == 3:
                        dicom_files = [f for f in dicom_files if not f.startswith('N')]

                if len(dicom_files) >= self.num_images_per_class:
                    selected_files = random.sample(dicom_files, self.num_images_per_class)
                else:
                    selected_files = dicom_files
                    
                # Group files into 3D volumes based on num_slices
                for i in range(0, len(selected_files), self.num_slices):
                    volume_files = selected_files[i:i + self.num_slices]
                    if len(volume_files) == self.num_slices:
                        volume_paths = [os.path.join(class_folder, f) for f in volume_files]
                        # Is a list of lists with 10 paths
                        self.image_volumes.append(volume_paths)
                        self.labels.append(class_label)

    def __len__(self):
        # lenght of outer list (amount of volumes)
        return len(self.image_volumes)
    
    def __getitem__(self, index, resize=(224,224)):
        volume_paths = self.image_volumes[index]
        volume_slices = []

        for path in volume_paths:
            dicom_image = pydicom.dcmread(path)
            image = dicom_image.pixel_array
            image = Image.fromarray(np.uint8(image))
            image = image.resize(resize)
            volume_slices.append(image)

        volume = np.stack([np.array(slice_img) for slice_img in volume_slices], axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[index]
        return volume, label
    
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

    def visualize_volumes(self, num_volumes=3):
        num_volumes = min(num_volumes, len(self.image_volumes))
        _, axes = plt.subplots(1, num_volumes, figsize=(15, 15))
        if num_volumes == 1:
            axes = [axes]
        for i in range(num_volumes):
            random_index = random.randint(0, len(self.image_volumes) - 1)
            volume, label = self.__getitem__(random_index)
            if isinstance(volume, torch.Tensor):
                volume = volume.squeeze().numpy()
            # Display middle slice of the 3D volume
            mid_slice = volume[len(volume) // 2]
            axes[i].imshow(mid_slice, cmap="gray")
            axes[i].set_title(f"Label: {self.classes[label]}")
            axes[i].axis("off")
        plt.show()

class DicomFineDataset3D(Dataset):
    def __init__(self, root_dir, classes, transform=None, num_slices=16):
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.num_slices = num_slices
        self.image_volumes = []
        self.labels = []

        file_groups = {}
        for file_name in os.listdir(root_dir):
            if file_name.endswith(".dcm"):
                prefix = file_name[0]
                if prefix in self.classes:
                    if prefix not in file_groups:
                        file_groups[prefix] = []
                    file_groups[prefix].append(os.path.join(root_dir, file_name))
        for prefix, files in file_groups.items():
            random.shuffle(files)
            for i in range(0, len(files), self.num_slices):
                volume_files = files[i: i + self.num_slices]
                if len(volume_files) == self.num_slices: #Only include complete volumes
                    self.image_volumes.append(volume_files)
                    self.labels.append(self.classes.index(prefix))

    def __len__(self):
        # length of outer list (amount of volumes)
        return len(self.image_volumes)
    
    def __getitem__(self, index, resize=(224,224)):
        volume_paths = self.image_volumes[index]
        volume_slices = []

        for path in volume_paths:
            dicom_image = pydicom.dcmread(path)
            image = dicom_image.pixel_array
            image = Image.fromarray(np.uint8(image))
            image = image.resize(resize)
            volume_slices.append(image)

        volume = np.stack([np.array(slice_img) for slice_img in volume_slices], axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[index]
        return volume, label
    
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

    def visualize_volumes(self, num_volumes=3):
        num_volumes = min(num_volumes, len(self.image_volumes))
        _, axes = plt.subplots(1, num_volumes, figsize=(15, 15))
        if num_volumes == 1:
            axes = [axes]
        for i in range(num_volumes):
            random_index = random.randint(0, len(self.image_volumes) - 1)
            volume, label = self.__getitem__(random_index)
            if isinstance(volume, torch.Tensor):
                volume = volume.squeeze().numpy()
            # Display middle slice of the 3D volume
            mid_slice = volume[len(volume) // 2]
            axes[i].imshow(mid_slice, cmap="gray")
            axes[i].set_title(f"Label: {self.classes[label]}")
            axes[i].axis("off")
        plt.show()

def display_data_loader_batch(data_loader, classes):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    num_images = min(len(images), 8)
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
        axes[i].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    plt.show()