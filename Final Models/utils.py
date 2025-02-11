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
        '''
        Initializes the dataset for the Coarse Level by selecting a specified number of DICOM images per class 
        and storing their file paths along with corresponding labels.

        Args:
            root_dir (str): The directory containing class subfolders with DICOM images.
            num_images_per_class (int): Number of images to select per class.
            classes (list): List of class names (subfolder names).
            transform (torchvision.transforms): A function for data transformations.
            scenario (int): Defines the selection process for non-nodule images.
        '''
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # If scenario 2 is selected only the non nodule images of the LIDC-IDRI dataset are included
        if scenario == 2:
            temp_folder = os.path.join(root_dir, "non-nodule")
            self.num_images_per_class = len([f for f in os.listdir(temp_folder) if f.endswith('.dcm') and f.startswith('N')])
        # For each class iterate through the specific subfolder in the folder location
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                # create a list of all paths to the dicom files in each subfolder
                dicom_files = [f for f in os.listdir(class_folder) if f.endswith('.dcm')]

                if class_name == "non-nodule":
                    # if scenario 2 is selected only include non nodule images from the LIDC-IDRI dataset
                    if scenario == 2:
                        dicom_files = [f for f in dicom_files if f.startswith('N')]
                    # if scenario 3 is selected only include non nodule images from the Lung-PET-CT-Dx dataset
                    elif scenario == 3:
                        dicom_files = [f for f in dicom_files if not f.startswith('N')]
                # if the number of images per class is set, then randomly subsample as many images as specified
                if len(dicom_files) >= self.num_images_per_class:
                    selected_files = random.sample(dicom_files, self.num_images_per_class)
                else:
                    selected_files = dicom_files
                for file_name in selected_files:
                    # create a list for all the dicom image paths and for all the corresponding labels
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(class_label)
        

    def __len__(self):
        # overwrite the length of the dataset with the amount of available image paths
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the dicom image is converted to a pillow image
        # and returned with its corrsponding label
        img_path = self.image_paths[index]
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    
    def get_labels(self):
        # function to return all the labels of the dataset
        return self.labels
    
    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [self.classes[label] for label in labels])
        plt.show()
    
    def visualize_images(self, num_images=5):
        # function to visualize a specified amount of images with their corresponding labels
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
        '''
        Initializes the dataset for the Fine Level by selecting a specified number of DICOM images per class 
        and storing their file paths along with corresponding labels.

        Args:
            root_dir (str): The directory containing class subfolders with DICOM images.
            classes (dictionary): Dictionary of classes with index as key and class name as value.
            transform (torchvision.transforms): A function for data transformations.
        '''
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Create a list with all image paths of the dicom files and a list with labels based on the prefix of the filename
        for file_name in os.listdir(root_dir):
            if file_name.endswith(".dcm"):
                prefix = file_name[0]
                if prefix in self.classes:
                    self.image_paths.append(os.path.join(root_dir, file_name))
                    self.labels.append(self.classes[prefix])

    def __len__(self):
        # overwrite the length of the dataset with the amount of available image paths
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the dicom image is converted to a pillow image
        # and returned with its corrsponding label
        img_path = self.image_paths[index]
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    
    def get_labels(self):
        # function to return all the labels of the dataset
        return self.labels
    
    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [list(self.classes.keys())[label] for label in labels])
        plt.show()

    def visualize_images(self, num_images=5):
        # function to visualize a specified amount of images with their corresponding labels
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
        '''
        Initializes the dataset for the Coarse Level for the 3D classifier by selecting a specified 
        number of DICOM images per class and storing their file paths along with corresponding labels.

        Args:
            root_dir (str): The directory containing class subfolders with DICOM images.
            num_images_per_class (int): Number of images to select per class.
            classes (list): List of class names (subfolder names).
            transform (torchvision.transforms): A function for data transformations.
            scenario (int): Defines the selection process for non-nodule images.
            num_slices (int): Number of slices as depth for the 3D CT volume.
        '''
        random.seed(41)
        self.root_dir = root_dir
        self.num_images_per_class = num_images_per_class
        self.classes = classes
        self.transform = transform
        self.num_slices = num_slices
        self.image_volumes = []
        self.labels = []
        # If scenario 2 is selected only the non nodule images of the LIDC-IDRI dataset are included
        if scenario == 2:
            temp_folder = os.path.join(root_dir, "non-nodule")
            self.num_images_per_class = len([f for f in os.listdir(temp_folder) if f.endswith('.dcm') and f.startswith('N')])
        # For each class iterate through the specific subfolder in the folder location
        for class_label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                dicom_files = sorted([f for f in os.listdir(class_folder) if f.endswith('.dcm')])

                # Scenario handling for non-nodule files
                if class_name == "non-nodule":
                    # if scenario 2 is selected only include non nodule images from the LIDC-IDRI dataset
                    if scenario == 2:
                        dicom_files = [f for f in dicom_files if f.startswith('N')]
                    # if scenario 3 is selected only include non nodule images from the Lung-PET-CT-Dx dataset
                    elif scenario == 3:
                        dicom_files = [f for f in dicom_files if not f.startswith('N')]
                # if the number of images per class is set, then randomly subsample as many images as specified
                if len(dicom_files) >= self.num_images_per_class:
                    selected_files = random.sample(dicom_files, self.num_images_per_class)
                else:
                    selected_files = dicom_files
                    
                # Group file paths into lists based on num_slices
                for i in range(0, len(selected_files), self.num_slices):
                    volume_files = selected_files[i:i + self.num_slices]
                    if len(volume_files) == self.num_slices:
                        volume_paths = [os.path.join(class_folder, f) for f in volume_files]
                        # create a list with a list for all the dicom image paths in the 3D volume and for all the corresponding labels
                        self.image_volumes.append(volume_paths)
                        self.labels.append(class_label)

    def __len__(self):
        # overwrite the length of the dataset with the amount of available image volumes
        return len(self.image_volumes)
    
    def __getitem__(self, index, resize=(224,224)):
        # if an instance of the dataset is retrieved, each slice of the 3D volume is converted to a pillow image
        # and returned with its corrsponding label
        volume_paths = self.image_volumes[index]
        volume_slices = []

        for path in volume_paths:
            dicom_image = pydicom.dcmread(path)
            image = dicom_image.pixel_array
            image = Image.fromarray(np.uint8(image))
            image = image.resize(resize)
            volume_slices.append(image)
        # stack the 2D slices to a 3D volume
        volume = np.stack([np.array(slice_img) for slice_img in volume_slices], axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[index]
        return volume, label
    
    def get_labels(self):
        # function to return all the labels of the dataset
        return self.labels
    
    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [self.classes[label] for label in labels])
        plt.show()

    def visualize_volumes(self, num_volumes=3):
        # function to visualize a specified amount of slices of the 3D volumes with their corresponding labels
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
    def __init__(self, root_dir, classes, transform=None, num_slices=16, final_evaluation=False):
        '''
        Initializes the dataset for the Fine Level for the 3D classifier by selecting a specified 
        number of DICOM images per class and storing their file paths along with corresponding labels.

        Args:
            root_dir (str): The directory containing class subfolders with DICOM images.
            classes (list): List of class names (subfolder names).
            transform (torchvision.transforms): A function for data transformations.
            num_slices (int): Number of slices as depth for the 3D CT volume.
            final_evaluation (bool): Boolean, if the daaset is used for the final evaluation of the model.
        '''
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.num_slices = num_slices
        self.image_volumes = []
        self.labels = []
        self.final_evaluation = final_evaluation
        self.image_paths = []
        self.labels_fine = []
        # Create a list with all image paths of the dicom files and a list with labels based on the prefix of the filename
        file_groups = {}
        for file_name in os.listdir(root_dir):
            if file_name.endswith(".dcm"):
                prefix = file_name[0]
                if prefix in self.classes:
                    if prefix not in file_groups:
                        file_groups[prefix] = []
                    file_groups[prefix].append(os.path.join(root_dir, file_name))
        # Group file paths into lists based on num_slices
        for prefix, files in file_groups.items():
            random.shuffle(files)
            for i in range(0, len(files), self.num_slices):
                volume_files = files[i: i + self.num_slices]
                if len(volume_files) == self.num_slices: #Only include complete volumes
                    # create a list with a list for all the dicom image paths in the 3D volume and for all the corresponding labels
                    self.image_volumes.append(volume_files)
                    self.labels.append(self.classes.index(prefix))
                    if self.final_evaluation:
                        # if the final evaluation is chosen, also a list with all file indices of the dicom files in the folder and 
                        # the corresponding fine label is created
                        self.image_paths.append([get_file_index(root_dir, slice_path.split("\\")[-1]) for slice_path in volume_files])
                        self.labels_fine.append([self.classes.index(prefix) for i in range(len(volume_files))])

    def __len__(self):
        # overwrite the length of the dataset with the amount of available image volumes
        return len(self.image_volumes)
    
    def __getitem__(self, index, resize=(224,224)):
        # if an instance of the dataset is retrieved, each slice of the 3D volume is converted to a pillow image
        # and returned with its corrsponding label
        volume_paths = self.image_volumes[index]
        volume_slices = []

        for path in volume_paths:
            dicom_image = pydicom.dcmread(path)
            image = dicom_image.pixel_array
            image = Image.fromarray(np.uint8(image))
            image = image.resize(resize)
            volume_slices.append(image)
        # stack the 2D slices to a 3D volume
        volume = np.stack([np.array(slice_img) for slice_img in volume_slices], axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[index]
        # if the final evaluation is chosen, the original and the fine labels are returned as well
        if self.final_evaluation:
            return volume, label, self.image_paths[index], self.labels_fine[index]
        else:
            return volume, label
    
    def get_labels(self):
        # function to return all the labels of the dataset
        return self.labels
    
    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [self.classes[label] for label in labels])
        plt.show()

    def visualize_volumes(self, num_volumes=3):
        # function to visualize a specified amount of slices of the 3D volumes with their corresponding labels
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
        '''
        Wrapper to apply the transformations to the dataset.

        Args:
            base_dataset (torch.utils.data.Dataset): The original dataset to wrap.
            transform (torchvision.transforms): A function for data transformations.
        '''
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the transforms will be applied to the image
        # and it will be returned together with its corresponding label
        sample, label = self.base_dataset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        # overwrite the length of the dataset with the length of the original unwrapped dataset
        return len(self.base_dataset)
    
class TransformDatasetBalanced(torch.utils.data.Dataset):
    def __init__(self, base_dataset, classes, transform=None, balance=True):
        '''
        Wrapper to apply the transformations to the dataset and balance it.

        Args:
            base_dataset (torch.utils.data.Dataset): The original dataset to wrap.
            classes (dict): Dictionary with class names as keys and index as values.
            transform (torchvision.transforms): A function for data transformations.
            balance (bool): Boolean, if the dataset should be balanced.
        '''
        self.base_dataset = base_dataset
        self.classes = classes
        self.transform = transform
        self.samples, self.labels = self.extract_data(base_dataset)
        # if balancing is chosen, the samples and labels are rebalanced
        if balance:
            self.samples, self.labels = self.balance_dataset(self.samples, self.labels)

    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the transforms will be applied to the image
        # and it will be returned together with its corresponding label
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
    def __len__(self):
        # overwrite the lenght function with the amount of samples
        return len(self.samples)

    def extract_data(self, dataset):
        # function to retrieve samples with their corresponding labels from a dataset and return them as a list
        samples, labels = [], []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples.append(sample)
            labels.append(label)
        return samples, labels

    def balance_dataset(self, samples, labels):
        # function to rebalance the dataset by upsampling the minority classes
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        new_samples = []
        new_labels = []
        for label, count in label_counts.items():
            indices = [i for i, l in enumerate(labels) if l == label]
            # randomly upsample the dataset by reselecting the indices of the samples
            additional_indices = np.random.choice(indices, max_count - count, replace=True)
            new_samples.extend([samples[i] for i in indices])
            new_samples.extend([samples[i] for i in additional_indices])
            new_labels.extend([labels[i] for i in indices])
            new_labels.extend([labels[i] for i in additional_indices])
        return new_samples, new_labels

    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [list(self.classes.keys())[label] for label in labels])
        plt.show()

    def visualize_images(self, num_images=5):
        # function to visualize a specified amount of slices of the 3D volumes with their corresponding labels
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
    '''
    Displays a batch of images from a DataLoader with their corresponding labels.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader to retrieve images and labels from.
        classes (list): A list of class names corresponding to the dataset labels.
    '''
    # retrieve a batch of the data loader and calculate the number of images to display (minimum of batch size or 8)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    num_images = min(len(images), 8)
    _, axes = plt.subplots(1, num_images, figsize=(15,15))
    if num_images == 1:
        axes = [axes]
    # create the plot for the images with their labels
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
    '''
    Displays a batch of images from a DataLoader with 3D volumes with their corresponding labels.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader to retrieve images and labels from.
        classes (list): A list of class names corresponding to the dataset labels.
    '''
    # retrieve a batch of the data loader and calculate the number of images to display (minimum of batch size or 8)
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
        if image.ndim == 4: 
            middle_slice = image[:, image.shape[1] // 2, :, :] 
        elif image.ndim == 3:  
            middle_slice = image[image.shape[0] // 2, :, :]  
        
        # For 2D representation, ensure channels are handled
        if middle_slice.ndim == 3 and middle_slice.shape[0] in [1, 3]:
            middle_slice = middle_slice.transpose(1, 2, 0)  
        elif middle_slice.ndim == 2: 
            middle_slice = middle_slice  

        # Normalize to [0, 1] for visualization
        middle_slice = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min())

        # Display the image
        axes[i].imshow(middle_slice, cmap="gray")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def hash_image(image):
    '''
    Hashes an image using SHA256 for comparison.

    Args:
        image (numpy.ndarray): The image to hash.
    '''
    image_bytes = image.tobytes()  # Convert image to bytes
    return hashlib.sha256(image_bytes).hexdigest()

def find_overlapping_images(train_dataset, test_dataset, logging=True):
    '''
    Checks if images in the training dataset overlap with the test dataset.
    Args:
        train_dataset (dataset): The training dataset.
        test_dataset (dataset): The test dataset.
        logging (bool): Boolean, whether the results should be logged.
    '''
    # Extract and hash all train images
    train_hashes = {}
    test_indices = []
    for idx, (image, _) in enumerate(train_dataset):
        # Convert to numpy if it is a tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        train_hashes[hash_image(image)] = idx

    # Check test images against train hashes
    overlaps = []
    for test_idx, (image, _) in enumerate(test_dataset):
        # Convert to numpy if it is a tensor
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
    '''
    Hashes an 3D volume using SHA256 for comparison.

    Args:
        image (numpy.ndarray): The 3D volume to hash.
    '''
    if image.ndim == 3:
        image = image.transpose(1,2,0)
    image_bytes = image.tobytes()
    return hashlib.sha256(image_bytes).hexdigest()

def find_overlapping_images_3d(train_dataset, test_dataset, logging=True):
    '''
    Checks if images in the training dataset overlap with the test dataset.
    Args:
        train_dataset (dataset): The training dataset.
        test_dataset (dataset): The test dataset.
        logging (bool): Boolean, whether the results should be logged.
    '''
    train_hashes = {}
    test_indices = []
    # Extract and hash all train images
    for idx, (image, _) in enumerate(train_dataset):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        train_hashes[hash_image(image)] = idx
    overlaps = []
    # Check test images against train hashes
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
    '''
    Removes overlapping images from a dataset by excluding specific indices.

    Args:
        dataset (Dataset): The original PyTorch dataset.
        overlapping_indices (list): A list of indices that should be removed.
    '''
    indices_to_remove = set(overlapping_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in indices_to_remove]
    return torch.utils.data.Subset(dataset, remaining_indices)

class TensorFolderDataset(Dataset):
    def __init__(self, folder_path):
        '''
        Loads all pt files from a folder as a dataset.

        Args:
            folder_path (str): The path to the folder containing `.pt` tensor files.
        '''
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        # overwrite the lenght function with the amount of pt files in the list
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # for each index return the image with the corresponding label from the loaded pt file
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        data = torch.load(file_path)
        return data['image'], data['label']   
   
class DICOMFlatDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, scenario=1, balance_n=True):
        '''
        Initializes the dataset for the Flat classification by selecting a specified number of DICOM images per class 
        and storing their file paths along with corresponding labels.

        Args:
            root_dir (str): The directory containing class subfolders with DICOM images.
            classes (list): List of class names (subfolder names).
            transform (torchvision.transforms): A function for data transformations.
            scenario (int): Defines the selection process for non-nodule images.
            balance_n (bool): Boolean, if the minority classes should be upsampled.
        '''
        random.seed(41)
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # iterate through each subfolder in the location (corresponding to the 5 classes of the flat classification problem)
        for folder in os.listdir(root_dir):
            for file_name in os.listdir(os.path.join(root_dir, folder)):
                if file_name.endswith(".dcm"):
                    prefix = file_name[0]
                    if folder == "non-nodule":
                        # if scenario 2 is selected only include non nodule images from the LIDC-IDRI dataset
                        if scenario == 2 and prefix != "N":
                            continue
                        # if scenario 3 is selected only include non nodule images from the Lung-PET-CT-Dx dataset
                        if scenario == 3 and prefix == "N":
                            continue
                        prefix = "N"
                    if prefix in self.classes:
                        file_name = os.path.join(root_dir, folder, file_name)
                        # create a list for all the dicom image paths and for all the corresponding labels
                        self.image_paths.append(file_name)
                        self.labels.append(self.classes[prefix])
        # execute the re-balancing
        if balance_n:
            self._balance_class_n()

    def __len__(self):
        # overwrite the length of the dataset with the amount of available image paths
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the dicom image is converted to a pillow image
        # and returned with its corrsponding label
        img_path = self.image_paths[index]
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    
    def _balance_class_n(self):
        # function for upsampling the minority classes
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
        # function to return all the labels of the dataset
        return self.labels
    
    def display_label_distribution(self):
        # function to visualize the label distribution in the dataset as a barchart
        label_counts = Counter(self.labels)
        labels, counts = zip(*label_counts.items())
        plt.bar(labels, counts)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.xticks(labels, [list(self.classes.keys())[label] for label in labels])
        plt.show()

    def visualize_images(self, num_images=5):
        # function to visualize a specified amount of images with their corresponding labels
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

class TensorFolderDatasetFinal(Dataset):
    def __init__(self, folder_path, depth=16):
        '''
        Creates a dataset from all pt files within a specified folder.

        Args:
            folder_path (str): Path to the folder containing pt files.
            depth (int): Depth for slice_paths and labels_fine when not available.
        '''
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        self.depth = depth

    def __len__(self):
         # overwrite the length of the dataset with the amount of files
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # for each index return the 3D volume with the corresponding coarse label and the paths to the original 2D slices with the fine labels
        slice_paths, labels_fine = None, None
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        data = torch.load(file_path)
        if "slice_paths" in data:
            slice_paths = data['slice_paths']
            labels_fine = data['labels_fine']
        else:
            slice_paths = torch.Tensor([0 for i in range(self.depth)])
            labels_fine = torch.Tensor([0 for i in range(self.depth)])
        return data['image'], data['label'], slice_paths, labels_fine
    
class TransformDatasetFinal(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        '''
        Wrapper to apply the transformations to the dataset.

        Args:
            base_dataset (torch.utils.data.Dataset): The original dataset to wrap.
            transform (torchvision.transforms): A function for data transformations.
        '''
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the transforms will be applied to the image
        # and it will be returned together with its corresponding label
        sample, label, slice_paths, labels_fine = self.base_dataset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label, slice_paths, labels_fine

    def __len__(self):
        # overwrite the length of the dataset with the length of the original unwrapped dataset
        return len(self.base_dataset)
    
class TensorFolderDatasetFinalFlat(Dataset):
    def __init__(self, tensor_folder_path, dicom_folder_path):
        '''
        Loads all pt files from a folder as a dataset for the final evaluation.

        Args:
            tensor_folder_path (str): The path to the folder containing `.pt` tensor files.
            dicom_folder_path (str): The path to the dicom files.
        '''
        self.tensor_folder_path = tensor_folder_path
        self.dicom_folder_path = dicom_folder_path
        self.indices = []
        self.labels = []
        # iterate through the pt files in the folder and store all paths to the slices as well as the fine labels in a list
        for pt_file in os.listdir(tensor_folder_path):
            temp_pt_file = torch.load(os.path.join(tensor_folder_path, pt_file))
            for index, value in zip(temp_pt_file['slice_paths'], temp_pt_file['label_fine']):
                key = sorted(os.listdir(dicom_folder_path))[index]
                self.indices.append(key)
                self.labels.append(value)

    def __len__(self):
        # overwrite the lenght function with the amount of slice paths
        return len(self.indices)
    
    def __getitem__(self, idx):
        # if an instance of the dataset is retrieved, the dicom image is converted to a pillow image
        # and returned with its corrsponding label
        key = self.indices[idx]
        dicom_image = pydicom.dcmread(os.path.join(self.dicom_folder_path, key))
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        return image, self.labels[idx]
    
class TransformDatasetFinalFlat(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        '''
        Wrapper to apply the transformations to the dataset.

        Args:
            base_dataset (torch.utils.data.Dataset): The original dataset to wrap.
            transform (torchvision.transforms): A function for data transformations.
        '''
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        # if an instance of the dataset is retrieved, the transforms will be applied to the image
        # and it will be returned together with its corresponding label
        sample, label_fine = self.base_dataset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label_fine

    def __len__(self):
        # overwrite the length of the dataset with the length of the original unwrapped dataset
        return len(self.base_dataset)
    
def get_file_index(folder_path, target_file):
    '''
    Retrieves the index of a target file in a sorted list of files from a given folder.

    Args:
        folder_path (str): The path to the folder containing the files.
        target_file (str): The name of the file whose index needs to be found.
    '''
    try:
        file_list = sorted(os.listdir(folder_path))
        index = file_list.index(target_file)
        return index
    except ValueError:
        return -1
    
def save_images_2D(file_dir, output_dir):
    '''
    Processes and saves 2D image slices from 3D tensor volumes stored in `.pt` files.

    Args:
        file_dir (str): Directory containing `.pt` files with 3D image volumes.
        output_dir (str): Directory where the extracted 2D slices will be saved.
    '''
    # load the pt files from the folder and retrieve the image and the coarse as well as the fine label
    for i, item in enumerate(os.listdir(file_dir)):
        if item != "2D":
            file_path = os.path.join(file_dir, item)
            data = torch.load(file_path)
            volume = data['image']
            label = data['label']
            label_fine = data['label_fine']
            if volume.shape[0] == 1:
                volume = volume.squeeze(0)
            if isinstance(volume, torch.Tensor):
                volume = volume.cpu().numpy()
            # iterate through the different slices of the 3D volume and safe them as pt file with the coarse and fine label
            for j in range(volume.shape[0]):
                slice_image = volume[j]
                slice_data = {
                    'image': slice_image,
                    'label': label,
                    'label_fine': label_fine
                }
                slice_file_path = os.path.join(output_dir, f"tensor_{i}_slice_{j}_label_{label}_label_fine_{label_fine}.pt")
                torch.save(slice_data, slice_file_path)