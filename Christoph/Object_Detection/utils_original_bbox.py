import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import hashlib
from torchvision import transforms

from PIL import Image
from collections import Counter
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data


class DICOMCoarseDataset(Dataset):
    def __init__(self, root_dir, num_images_per_class, classes, bbox_coord, transform=None, scenario=1):
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.bbox_coord = bbox_coord
        self.image_paths = []
        self.labels = []
        self.bbox = []

        # if scenario == 2:
        #     temp_folder = os.path.join(root_dir, "non-nodule")
        #     self.num_images_per_class = len([f for f in os.listdir(temp_folder) if f.endswith('.dcm') and f.startswith('N')])

        # Loop through each class and process its directory.
        for class_label, class_name in enumerate(self.classes):
            # Path to the current class folder.
            class_folder = os.path.join(root_dir, class_name)

            # Check if the class folder exists.
            if os.path.isdir(class_folder):
                # List all files in the folder that end with '.dcm' (DICOM files).
                dicom_files = [f for f in os.listdir(
                    class_folder) if f.endswith('.dcm')]

                # Special filtering for the "non-nodule" class based on the scenario.
                if class_name == "non-nodule":
                    if scenario == 2:
                        # Scenario 2: Keep only files starting with 'N'.
                        dicom_files = [
                            f for f in dicom_files if f.startswith('N')]
                    elif scenario == 3:
                        # Scenario 3: Exclude files starting with 'N'.
                        dicom_files = [
                            f for f in dicom_files if not f.startswith('N')]

                # Randomly sample the specified number of images or use all if fewer exist.
                if len(dicom_files) >= self.num_images_per_class:
                    selected_files = random.sample(
                        dicom_files, self.num_images_per_class)
                else:
                    # Use all available files if insufficient.
                    selected_files = dicom_files

                # Add the selected file paths and their corresponding labels to the dataset.
                for file_name in selected_files:
                    dicom_data = pydicom.dcmread(
                        os.path.join(class_folder, file_name))
                    uid = dicom_data.SOPInstanceUID  # Extract UID from DICOM metadata

                    # Look for matching bounding box coordinates in bbox_coord DataFrame
                    matching_row = self.bbox_coord[self.bbox_coord['UID_Annotation'] == uid]

                    if not matching_row.empty:
                        # Append bounding box coordinates if found
                        bbox_data = {
                            'xmin': matching_row.iloc[0]['xmin'],
                            'ymin': matching_row.iloc[0]['ymin'],
                            'xmax': matching_row.iloc[0]['xmax'],
                            'ymax': matching_row.iloc[0]['ymax']
                        }

                    else:
                        # Indicate no bounding box for this image
                        bbox_data = {

                        }

                    # Full file path.
                    self.image_paths.append(
                        os.path.join(class_folder, file_name))
                    # Numerical label corresponding to the class.
                    self.labels.append(class_label)
                    # Add bbox coordinates to the list
                    self.bbox.append(bbox_data)

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
        bbox = self.bbox[index]
        return image, label, bbox

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
        _, axes = plt.subplots(1, num_images, figsize=(15, 15))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            random_index = random.randint(0, len(self.image_paths)-1)
            image, label, bbox = self.__getitem__(random_index)
            if isinstance(image, torch.Tensor):
                image = image.squeeze().numpy()
            axes[i].imshow(image, cmap="gray")

            # Wenn eine Bounding Box vorhanden ist (nicht [-1, -1, -1, -1]), zeichne sie
            if bbox and all(coord != -1 for coord in bbox.values()):
                axes[i].add_patch(plt.Rectangle(
                    (bbox['xmin'], bbox['ymin']),
                    bbox['xmax'] - bbox['xmin'],
                    bbox['ymax'] - bbox['ymin'],
                    fill=False, color='red', linewidth=2  # Rechteck in rot
                ))
            axes[i].set_title(f"Label: {self.classes[label]}")
            axes[i].axis("off")
        plt.show()


class CustomTransform:
    def __init__(self, image_size):
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_size = image_size

    def __call__(self, sample):
        image, label, bbox = sample['image'], sample['bbox'], sample['label']
        # print(type(bbox))
        # Get original image dimensions
        orig_width, orig_height = image.size  # PIL image

        # Apply image transformation
        image = self.image_transform(image)

        if bbox['xmin'] == -1 and bbox['ymin'] == -1 and bbox['xmax'] == -1 and bbox['ymax'] == -1:
            # Set scaled_bbox to -1, -1, -1, -1 to indicate no bounding box
            scaled_bbox = (bbox['xmin'],bbox['ymin'],  bbox['xmax'], bbox['ymax'])
        else:
            x_min, y_min, x_max, y_max = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            x_min = x_min * self.image_size / orig_width
            y_min = y_min * self.image_size / orig_height
            x_max = x_max * self.image_size / orig_width
            y_max = y_max * self.image_size / orig_height
            scaled_bbox = (x_min, y_min, x_max, y_max)

        return {'image': image, 'label': label, 'bbox': scaled_bbox}

class TransformBboxDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, bbox, label = self.dataset[index]

        # Wenn eine Transformation vorhanden ist, wende sie an
        if self.transform:
            sample = {'image': image, 'bbox': bbox, 'label': label}
            sample = self.transform(sample)

        image = sample['image']
        label = sample['label']
        bbox = sample['bbox']

        return image, label, bbox


# class DICOMFineDataset(Dataset):
#     def __init__(self, root_dir, classes, transform=None):
#         random.seed(41)
#         self.root_dir = root_dir
#         self.classes = classes
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []

#         for file_name in os.listdir(root_dir):
#             if file_name.endswith(".dcm"):
#                 prefix = file_name[0]
#                 if prefix in self.classes:
#                     self.image_paths.append(os.path.join(root_dir, file_name))
#                     self.labels.append(self.classes[prefix])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         dicom_image = pydicom.dcmread(img_path)
#         image = dicom_image.pixel_array
#         image = Image.fromarray(np.uint8(image))
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels[index]
#         return image, label

#     def get_labels(self):
#         return self.labels

#     def display_label_distribution(self):
#         label_counts = Counter(self.labels)
#         labels, counts = zip(*label_counts.items())
#         plt.bar(labels, counts)
#         plt.xlabel("Label")
#         plt.ylabel("Count")
#         plt.title("Label Distribution")
#         plt.xticks(labels, [list(self.classes.keys())[label]
#                    for label in labels])
#         plt.show()

#     def visualize_images(self, num_images=5):
#         num_images = min(num_images, len(self.image_paths))
#         _, axes = plt.subplots(1, num_images, figsize=(15, 15))
#         if num_images == 1:
#             axes = [axes]
#         for i in range(num_images):
#             random_index = random.randint(0, len(self.image_paths) - 1)
#             image, label = self.__getitem__(random_index)
#             if isinstance(image, torch.Tensor):
#                 image = image.squeeze().numpy()
#             axes[i].imshow(image, cmap="gray")
#             axes[i].set_title(f"Label: {list(self.classes.keys())[label]}")
#             axes[i].axis("off")
#         plt.show()


# class DicomCoarseDataset3D(Dataset):
#     def __init__(self, root_dir, num_images_per_class, classes, transform=None, scenario=1, num_slices=16):
#         random.seed(41)
#         self.root_dir = root_dir
#         self.num_images_per_class = num_images_per_class
#         self.classes = classes
#         self.transform = transform
#         self.num_slices = num_slices
#         self.image_volumes = []
#         self.labels = []

#         if scenario == 2:
#             temp_folder = os.path.join(root_dir, "non-nodule")
#             self.num_images_per_class = len([f for f in os.listdir(
#                 temp_folder) if f.endswith('.dcm') and f.startswith('N')])

#         for class_label, class_name in enumerate(self.classes):
#             class_folder = os.path.join(root_dir, class_name)
#             if os.path.isdir(class_folder):
#                 dicom_files = sorted(
#                     [f for f in os.listdir(class_folder) if f.endswith('.dcm')])

#                 # Scenario handling for non-nodule files
#                 if class_name == "non-nodule":
#                     if scenario == 2:
#                         dicom_files = [
#                             f for f in dicom_files if f.startswith('N')]
#                     elif scenario == 3:
#                         dicom_files = [
#                             f for f in dicom_files if not f.startswith('N')]

#                 if len(dicom_files) >= self.num_images_per_class:
#                     selected_files = random.sample(
#                         dicom_files, self.num_images_per_class)
#                 else:
#                     selected_files = dicom_files

#                 # Group files into 3D volumes based on num_slices
#                 for i in range(0, len(selected_files), self.num_slices):
#                     volume_files = selected_files[i:i + self.num_slices]
#                     if len(volume_files) == self.num_slices:
#                         volume_paths = [os.path.join(
#                             class_folder, f) for f in volume_files]
#                         # Is a list of lists with 10 paths
#                         self.image_volumes.append(volume_paths)
#                         self.labels.append(class_label)

#     def __len__(self):
#         # lenght of outer list (amount of volumes)
#         return len(self.image_volumes)

#     def __getitem__(self, index, resize=(224, 224)):
#         volume_paths = self.image_volumes[index]
#         volume_slices = []

#         for path in volume_paths:
#             dicom_image = pydicom.dcmread(path)
#             image = dicom_image.pixel_array
#             image = Image.fromarray(np.uint8(image))
#             image = image.resize(resize)
#             volume_slices.append(image)

#         volume = np.stack([np.array(slice_img)
#                           for slice_img in volume_slices], axis=0)
#         if self.transform:
#             volume = self.transform(volume)
#         label = self.labels[index]
#         return volume, label

#     def get_labels(self):
#         return self.labels

#     def display_label_distribution(self):
#         label_counts = Counter(self.labels)
#         labels, counts = zip(*label_counts.items())
#         plt.bar(labels, counts)
#         plt.xlabel("Label")
#         plt.ylabel("Count")
#         plt.title("Label Distribution")
#         plt.xticks(labels, [self.classes[label] for label in labels])
#         plt.show()

#     def visualize_volumes(self, num_volumes=3):
#         num_volumes = min(num_volumes, len(self.image_volumes))
#         _, axes = plt.subplots(1, num_volumes, figsize=(15, 15))
#         if num_volumes == 1:
#             axes = [axes]
#         for i in range(num_volumes):
#             random_index = random.randint(0, len(self.image_volumes) - 1)
#             volume, label = self.__getitem__(random_index)
#             if isinstance(volume, torch.Tensor):
#                 volume = volume.squeeze().numpy()
#             # Display middle slice of the 3D volume
#             mid_slice = volume[len(volume) // 2]
#             axes[i].imshow(mid_slice, cmap="gray")
#             axes[i].set_title(f"Label: {self.classes[label]}")
#             axes[i].axis("off")
#         plt.show()


# class DicomFineDataset3D(Dataset):
#     def __init__(self, root_dir, classes, transform=None, num_slices=16):
#         random.seed(41)
#         self.root_dir = root_dir
#         self.classes = classes
#         self.transform = transform
#         self.num_slices = num_slices
#         self.image_volumes = []
#         self.labels = []

#         file_groups = {}
#         for file_name in os.listdir(root_dir):
#             if file_name.endswith(".dcm"):
#                 prefix = file_name[0]
#                 if prefix in self.classes:
#                     if prefix not in file_groups:
#                         file_groups[prefix] = []
#                     file_groups[prefix].append(
#                         os.path.join(root_dir, file_name))
#         for prefix, files in file_groups.items():
#             random.shuffle(files)
#             for i in range(0, len(files), self.num_slices):
#                 volume_files = files[i: i + self.num_slices]
#                 if len(volume_files) == self.num_slices:  # Only include complete volumes
#                     self.image_volumes.append(volume_files)
#                     self.labels.append(self.classes.index(prefix))

#     def __len__(self):
#         # length of outer list (amount of volumes)
#         return len(self.image_volumes)

#     def __getitem__(self, index, resize=(224, 224)):
#         volume_paths = self.image_volumes[index]
#         volume_slices = []

#         for path in volume_paths:
#             dicom_image = pydicom.dcmread(path)
#             image = dicom_image.pixel_array
#             image = Image.fromarray(np.uint8(image))
#             image = image.resize(resize)
#             volume_slices.append(image)

#         volume = np.stack([np.array(slice_img)
#                           for slice_img in volume_slices], axis=0)
#         if self.transform:
#             volume = self.transform(volume)
#         label = self.labels[index]
#         return volume, label

#     def get_labels(self):
#         return self.labels

#     def display_label_distribution(self):
#         label_counts = Counter(self.labels)
#         labels, counts = zip(*label_counts.items())
#         plt.bar(labels, counts)
#         plt.xlabel("Label")
#         plt.ylabel("Count")
#         plt.title("Label Distribution")
#         plt.xticks(labels, [self.classes[label] for label in labels])
#         plt.show()

#     def visualize_volumes(self, num_volumes=3):
#         num_volumes = min(num_volumes, len(self.image_volumes))
#         _, axes = plt.subplots(1, num_volumes, figsize=(15, 15))
#         if num_volumes == 1:
#             axes = [axes]
#         for i in range(num_volumes):
#             random_index = random.randint(0, len(self.image_volumes) - 1)
#             volume, label = self.__getitem__(random_index)
#             if isinstance(volume, torch.Tensor):
#                 volume = volume.squeeze().numpy()
#             # Display middle slice of the 3D volume
#             mid_slice = volume[len(volume) // 2]
#             axes[i].imshow(mid_slice, cmap="gray")
#             axes[i].set_title(f"Label: {self.classes[label]}")
#             axes[i].axis("off")
#         plt.show()


# class TransformDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dataset, transform=None):
#         self.base_dataset = base_dataset
#         self.transform = transform

#     def __getitem__(self, index):
#         sample, label = self.base_dataset[index]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, label

#     def __len__(self):
#         return len(self.base_dataset)


# class TransformDatasetBalanced(torch.utils.data.Dataset):
#     def __init__(self, base_dataset, classes, transform=None, balance=True):
#         self.base_dataset = base_dataset
#         self.classes = classes
#         self.transform = transform
#         self.samples, self.labels = self.extract_data(base_dataset)
#         if balance:
#             self.samples, self.labels = self.balance_dataset(
#                 self.samples, self.labels)

#     def __getitem__(self, index):
#         sample = self.samples[index]
#         label = self.labels[index]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, label

#     def __len__(self):
#         return len(self.samples)

#     def extract_data(self, dataset):
#         samples, labels = [], []
#         for i in range(len(dataset)):
#             sample, label = dataset[i]
#             samples.append(sample)
#             labels.append(label)
#         return samples, labels

#     def balance_dataset(self, samples, labels):
#         label_counts = Counter(labels)
#         max_count = max(label_counts.values())
#         new_samples = []
#         new_labels = []
#         for label, count in label_counts.items():
#             indices = [i for i, l in enumerate(labels) if l == label]
#             additional_indices = np.random.choice(
#                 indices, max_count - count, replace=True)
#             new_samples.extend([samples[i] for i in indices])
#             new_samples.extend([samples[i] for i in additional_indices])
#             new_labels.extend([labels[i] for i in indices])
#             new_labels.extend([labels[i] for i in additional_indices])
#         return new_samples, new_labels

#     def display_label_distribution(self):
#         label_counts = Counter(self.labels)
#         labels, counts = zip(*label_counts.items())
#         plt.bar(labels, counts)
#         plt.xlabel("Label")
#         plt.ylabel("Count")
#         plt.title("Label Distribution")
#         plt.xticks(labels, [list(self.classes.keys())[label]
#                    for label in labels])
#         plt.show()

#     def visualize_images(self, num_images=5):
#         num_images = min(num_images, len(self.samples))
#         _, axes = plt.subplots(1, num_images, figsize=(15, 15))
#         if num_images == 1:
#             axes = [axes]
#         for i in range(num_images):
#             random_index = random.randint(0, len(self.samples) - 1)
#             image, label = self.__getitem__(random_index)
#             if isinstance(image, torch.Tensor):
#                 image = image.permute(1, 2, 0).cpu().numpy()
#             image = (image - image.min()) / (image.max() - image.min())
#             axes[i].imshow(image, cmap="gray")
#             axes[i].set_title(f"Label: {list(self.classes.keys())[label]}")
#             axes[i].axis("off")
#         plt.show()


def display_data_loader_batch(data_loader, classes):
    data_iter = iter(data_loader)
    images, labels, bbox = next(data_iter)
    num_images = min(len(images), 8)
    _, axes = plt.subplots(1, num_images, figsize=(15, 15))
    if num_images == 1:
        axes = [axes]
    for i in range(num_images):
        image = images[i].cpu()
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
        # Normalize and adjust dimensions for display
        if image.ndim == 3 and image.shape[-1] == 1:
            # Remove the channel dimension for grayscale
            image = image.squeeze(axis=-1)
        elif image.ndim == 2:
            image = image  # Grayscale images should remain 2D
        axes[i].imshow(image, cmap="gray")
        current_bbox = [tensor[i].item() for tensor in bbox]
        xmin, ymin, xmax, ymax = current_bbox[0], current_bbox[1], current_bbox[2], current_bbox[3]
        if  [xmin, ymin, xmax, ymax] != [-1, -1, -1, -1]:
            axes[i].add_patch(plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False, color='red', linewidth=2  # Rechteck in rot
                ))
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    plt.show()


# def display_data_loader_batch_3d(data_loader, classes):
#     data_iter = iter(data_loader)
#     images, labels = next(data_iter)
#     # Number of images to display
#     num_images = min(len(images), 8)
#     _, axes = plt.subplots(1, num_images, figsize=(15, 15))
#     if num_images == 1:
#         axes = [axes]

#     for i in range(num_images):
#         # Move image to CPU and convert to NumPy
#         image = images[i].cpu().numpy()

#         # Handle 3D images by selecting the middle slice along the depth axis
#         if image.ndim == 4:  # Shape: (C, D, H, W)
#             # Select middle slice along depth
#             middle_slice = image[:, image.shape[1] // 2, :, :]
#         elif image.ndim == 3:  # Shape: (D, H, W)
#             # Middle slice for grayscale
#             middle_slice = image[image.shape[0] // 2, :, :]

#         # For 2D representation, ensure channels are handled
#         # Shape: (C, H, W)
#         if middle_slice.ndim == 3 and middle_slice.shape[0] in [1, 3]:
#             middle_slice = middle_slice.transpose(
#                 1, 2, 0)  # Convert to (H, W, C)
#         elif middle_slice.ndim == 2:  # Shape: (H, W)
#             middle_slice = middle_slice  # Grayscale images remain as-is

#         # Normalize to [0, 1] for visualization
#         middle_slice = (middle_slice - middle_slice.min()) / \
#             (middle_slice.max() - middle_slice.min())

#         # Display the image
#         axes[i].imshow(middle_slice, cmap="gray")
#         axes[i].set_title(f"Label: {classes[labels[i].item()]}")
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()


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


def find_overlapping_images(train_dataset, test_dataset):
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

    print(f"Found {len(overlaps)} overlapping images")
    for train_idx, test_idx in overlaps:
        print(f"Train index: {train_idx}, Test index: {test_idx}")
