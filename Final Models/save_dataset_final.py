import torch
import os

from torch.utils.data import random_split
from utils import DicomFineDataset3D

def save_subset_locally(save_folder, subset):
    os.makedirs(save_folder, exist_ok=True)
    for i, (image, label, slice_paths, labels_fine) in enumerate(subset):
        tensor_file = f'{save_folder}/tensor_{i}_label_coarse_0_label_fine_{label}.pt'
        torch.save({'image': image, 'label': 0, 'slice_paths': torch.Tensor(slice_paths), 'labels_fine': torch.Tensor(labels_fine)}, tensor_file)
    print(f'Subset saved to {save_folder}')

if __name__ == "__main__":
    save_folder = f"./data/final"
    train_ratio = 0.6

    BASE_DIR = "C:/Users/Dominik Hahn/OneDrive/Studium/Master/Masterarbeit/Daten/nodule"
    classes = ["A", "B", "E", "G"]

    dataset = DicomFineDataset3D(root_dir=BASE_DIR, classes=classes, final_evaluation=True)
    train_size = int(train_ratio * len(dataset))
    val_size = int(((1-train_ratio)/2) * len(dataset)) 
    test_size = len(dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(dataset, [train_size, val_size, test_size])
    save_subset_locally(os.path.join(save_folder, "test_fine"), test_indices)