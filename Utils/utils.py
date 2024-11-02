import os
import pandas as pd


class Utils():
    def __init__(self) -> None:        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

    def create_filepath_dict(self):
        filepath_dict, all_file_dict = dict(), dict()
        all_folders = [name for name in os.listdir(self.DATA_DIR) if os.path.isdir(os.path.join(self.DATA_DIR, name))]
        for folder in all_folders:
            current_folder = os.path.join(self.DATA_DIR, folder)
            filepath_dict[folder] = [name for name in os.listdir(current_folder) if os.path.isdir(os.path.join(current_folder, name))]
        for folder, subfolders in filepath_dict.items():
            all_file_dict[folder] = list()
            for subfolder in subfolders:
                current_list = list()
                for file in os.listdir(os.path.join(self.DATA_DIR, folder, subfolder)):
                    current_list.append(file)
                all_file_dict[folder].append({subfolder: current_list})
        return all_file_dict
    
    def create_labels(self):
        #'A' were diagnosed with Adenocarcinoma, 
        #'B' with Small Cell Carcinoma, 
        #'E' with Large Cell Carcinoma,
        #'G' with Squamous Cell Carcinoma.
        folder_list, label_list = list(), list()
        for folder in [name for name in os.listdir(self.DATA_DIR) if os.path.isdir(os.path.join(self.DATA_DIR, name))]:
            cleaned_folder_name = folder.replace("Lung_Dx-", "")
            cleaned_folder_name = cleaned_folder_name[0]
            folder_list.append(folder)
            label_list.append(cleaned_folder_name)
        df = pd.DataFrame({"Label": label_list, "Folder": folder_list})
        df.to_csv("./data/labels.csv")
