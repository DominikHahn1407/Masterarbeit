import torch
import os

from torch.utils.data import random_split
from utils import DICOMCoarseDataset, DicomCoarseDataset3D, DICOMFineDataset, DicomFineDataset3D

def save_subset_locally(save_folder, subset):
    os.makedirs(save_folder, exist_ok=True)
    for i, (image, label) in enumerate(subset):
        tensor_file = f'{save_folder}/tensor_{i}_label_{label}.pt'
        torch.save({'image': image, 'label': label}, tensor_file)
    print(f'Subset saved to {save_folder}')

if __name__ == "__main__":
    scenario = 1
    train_ratio = 0.6

    BASE_DIR = "C:/Users/Dominik Hahn/OneDrive/Studium/Master/Masterarbeit/Daten"
    classes = ["nodule", "non-nodule"]

    dataset = DICOMCoarseDataset(root_dir=BASE_DIR, num_images_per_class=len(os.listdir(os.path.join(BASE_DIR, "nodule"))), classes=classes, scenario=scenario)
    train_size = int(train_ratio * len(dataset))
    val_size = int(((1-train_ratio)/2) * len(dataset)) 
    test_size = len(dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(dataset, [train_size, val_size, test_size])

    save_folder = "./data/scenario3/test"
    save_subset_locally(save_folder, test_indices)