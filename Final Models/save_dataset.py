import torch
import os

from torch.utils.data import random_split
from utils import DicomFineDataset3D
def save_subset_locally(save_folder, subset):
    # Saves a subset of image tensors and associated labels locally in a specified folder.
    # This method is used for the creation of fixed training, validation and test sets
    os.makedirs(save_folder, exist_ok=True)
    for i, (image, label) in enumerate(subset):
        tensor_file = f'{save_folder}/tensor_{i}_label_{label}.pt'
        torch.save({'image': image, 'label': label}, tensor_file)
    print(f'Subset saved to {save_folder}')

if __name__ == "__main__":
    scenario = 3
    save_folder = f"./data/scenario{scenario}_flat"
    train_ratio = 0.6

    BASE_DIR = "C:/Users/Dominik Hahn/OneDrive/Studium/Master/Masterarbeit/Daten"
    classes = ["A", "B", "E", "G", "N"]
    classes_dict = {value: index for index, value in enumerate(classes)}
    # Splitting the Dataset into train, val and test and saving it to the specified folder
    dataset = DICOMFlatDataset(root_dir=BASE_DIR, classes=classes_dict, scenario=scenario, balance_n=True)
    train_size = int(train_ratio * len(dataset))
    val_size = int(((1-train_ratio)/2) * len(dataset)) 
    test_size = len(dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(dataset, [train_size, val_size, test_size])
    save_subset_locally(os.path.join(save_folder, "test_coarse"), test_indices)