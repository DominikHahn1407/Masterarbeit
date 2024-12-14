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

        if scenario == 2:
            temp_folder = os.path.join(root_dir, "non-nodule")
            self.num_images_per_class = len([f for f in os.listdir(temp_folder) if f.endswith('.dcm') and f.startswith('N')])

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

        if scenario == 2:
            temp_folder = os.path.join(root_dir, "non-nodule")
            self.num_images_per_class = len([f for f in os.listdir(temp_folder) if f.endswith('.dcm') and f.startswith('N')])

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
    

class TransformDatasetBalanced(torch.utils.data.Dataset):
    def __init__(self, base_dataset, classes, transform=None, balance=True):
        self.base_dataset = base_dataset
        self.classes = classes
        self.transform = transform
        self.samples, self.labels = self.extract_data(base_dataset)
        if balance:
            self.samples, self.labels = self.balance_dataset(self.samples, self.labels)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
    def __len__(self):
        return len(self.samples)

    def extract_data(self, dataset):
        samples, labels = [], []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples.append(sample)
            labels.append(label)
        return samples, labels

    def balance_dataset(self, samples, labels):
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        new_samples = []
        new_labels = []
        for label, count in label_counts.items():
            indices = [i for i, l in enumerate(labels) if l == label]
            additional_indices = np.random.choice(indices, max_count - count, replace=True)
            new_samples.extend([samples[i] for i in indices])
            new_samples.extend([samples[i] for i in additional_indices])
            new_labels.extend([labels[i] for i in indices])
            new_labels.extend([labels[i] for i in additional_indices])
        return new_samples, new_labels

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
        num_images = min(num_images, len(self.samples))
        _, axes = plt.subplots(1, num_images, figsize=(15, 15))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            random_index = random.randint(0, len(self.samples) - 1)
            image, label = self.__getitem__(random_index)
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            axes[i].imshow(image, cmap="gray")
            axes[i].set_title(f"Label: {list(self.classes.keys())[label]}")
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
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    plt.show()

def display_data_loader_batch_3d(data_loader, classes):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    # Number of images to display
    num_images = min(len(images), 8)
    _, axes = plt.subplots(1, num_images, figsize=(15, 15))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        # Move image to CPU and convert to NumPy
        image = images[i].cpu().numpy()
        
        # Handle 3D images by selecting the middle slice along the depth axis
        if image.ndim == 4:  # Shape: (C, D, H, W)
            middle_slice = image[:, image.shape[1] // 2, :, :]  # Select middle slice along depth
        elif image.ndim == 3:  # Shape: (D, H, W)
            middle_slice = image[image.shape[0] // 2, :, :]  # Middle slice for grayscale
        
        # For 2D representation, ensure channels are handled
        if middle_slice.ndim == 3 and middle_slice.shape[0] in [1, 3]:  # Shape: (C, H, W)
            middle_slice = middle_slice.transpose(1, 2, 0)  # Convert to (H, W, C)
        elif middle_slice.ndim == 2:  # Shape: (H, W)
            middle_slice = middle_slice  # Grayscale images remain as-is

        # Normalize to [0, 1] for visualization
        middle_slice = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min())

        # Display the image
        axes[i].imshow(middle_slice, cmap="gray")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def hash_image(image):
    """
    Hashes an image using SHA256 for comparison.
    Args:
        image (numpy.ndarray): The image to hash.
    Returns:
        str: The hash of the image.
    """
    image_bytes = image.tobytes()  # Convert image to bytes
    return hashlib.sha256(image_bytes).hexdigest()

def find_overlapping_images(train_dataset, test_dataset, logging=True):
    """
    Checks if images in the training dataset overlap with the test dataset.
    Args:
        train_dataset: The training dataset.
        test_dataset: The test dataset.
    Returns:
        list: List of overlapping indices (train_idx, test_idx).
    """
    # Extract and hash all train images
    train_hashes = {}
    test_indices = []
    for idx, (image, _) in enumerate(train_dataset):
        # Convert to numpy if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        train_hashes[hash_image(image)] = idx

    # Check test images against train hashes
    overlaps = []
    for test_idx, (image, _) in enumerate(test_dataset):
        # Convert to numpy if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        test_hash = hash_image(image)
        if test_hash in train_hashes:
            overlaps.append((train_hashes[test_hash], test_idx))
    if logging:
        print(f"Found {len(overlaps)} overlapping images")
    for train_idx, test_idx in overlaps:
        if logging:
            print(f"Train index: {train_idx}, Test index: {test_idx}")
        test_indices.append(test_idx)
    return test_indices

def hash_image_3d(image):
    if image.ndim == 3:
        image = image.transpose(1,2,0)
    image_bytes = image.tobytes()
    return hashlib.sha256(image_bytes).hexdigest()

def find_overlapping_images_3d(train_dataset, test_dataset, logging=True):
    train_hashes = {}
    test_indices = []
    for idx, (image, _) in enumerate(train_dataset):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        train_hashes[hash_image(image)] = idx
    overlaps = []
    for test_idx, (image, _) in enumerate(test_dataset):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        test_hash = hash_image(image)
        if test_hash in train_hashes:
            overlaps.append((train_hashes[test_hash], test_idx))
    if logging:
        print(f"Found {len(overlaps)} overlapping images")
    for train_idx, test_idx in overlaps:
        if logging:
            print(f"Train index: {train_idx}, Test index: {test_idx}")
        test_indices.append(test_idx)
    return test_indices 

def remove_overlapping_images(dataset, overlapping_indices):
    indices_to_remove = set(overlapping_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in indices_to_remove]
    return torch.utils.data.Subset(dataset, remaining_indices)

class TensorFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        data = torch.load(file_path)
        return data['image'], data['label']                                  
    
class DICOMFlatDataset(Dataset):
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