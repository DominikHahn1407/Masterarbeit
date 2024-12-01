import os 
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from PIL import Image
from collections import Counter
from torch.utils.data import Dataset

# Definition der DICOMCoarseDataset-Klasse, die von der PyTorch Dataset-Basisklasse erbt
class DICOMPETDataset(Dataset):
    
    def __init__(self, df, transform=None, target_size=(512, 512)):
        """
        Args:
            df (pd.DataFrame): DataFrame mit Bildinformationen.
            transform (callable, optional): Transformationen, die auf das Bild angewendet werden sollen.
        """
        # DataFrame direkt speichern
        self.data = df
        self.transform = transform
        self.target_size = target_size
        self.class_names = ['Negativ', 'Positiv']
        self.label_mapping = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        # Gibt die Gesamtanzahl der Datensätze im Dataset zurück
        return len(self.data)

    def __getitem__(self, idx):
        # Dateipfad des Bildes basierend auf dem Index abrufen
        img_path = self.data.iloc[idx]['File Path']
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = Image.fromarray(np.uint8(image))
        
        # Originalbildgröße
        w_orig, h_orig = image.size
        
        # Bounding-Box-Koordinaten abrufen
        xmin = self.data.iloc[idx]['xmin']
        ymin = self.data.iloc[idx]['ymin']
        xmax = self.data.iloc[idx]['xmax']
        ymax = self.data.iloc[idx]['ymax']

        # Skalierung der Bounding Boxen
        if self.transform:
            image = self.transform(image)
            # Neue Bildgröße nach der Transformation
        w_new, h_new = self.target_size

        # Skalieren der Bounding Boxen
        xmin = xmin * (w_new / w_orig)
        ymin = ymin * (h_new / h_orig)
        xmax = xmax * (w_new / w_orig)
        ymax = ymax * (h_new / h_orig)

        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                
        # Label abrufen und in einen numerischen Index umwandeln
        label = self.data.iloc[idx]['Label']  # Hier ist das Label als String (z.B. 'cat')
        label_idx = self.label_mapping[label]  # Mapping des Labels auf den Index
        label = torch.tensor(label_idx)
        # One-Hot-Encoding mit torch.eye erstellen
        # label = torch.eye(len(self.class_names))[label_idx]  # Auswahl des richtigen One-Hot-Vektors
        

        # Bild, Bounding Box und Label zurückgeben
        return image,label,bbox
    
    
    def display_label_distribution(self):
        """
        Visualisiert die Verteilung der Labels im Dataset als Balkendiagramm.
        """
        # Zählt die Häufigkeit jedes Labels
        label_counts = Counter(self.data['Label'])

        # Labels und deren Häufigkeiten extrahieren
        labels, counts = zip(*label_counts.items())

        # Erstellen eines Balkendiagramms
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        
        # Achsen beschriften
        plt.xlabel('Label')
        plt.ylabel('Anzahl der Vorkommen')
        plt.title('Verteilung der Labels im Dataset')

        # Zeigt das Diagramm an
        plt.show()

    
    def visualize_images_with_bboxes(self, num_images=5):
        """
        Visualisiert eine Auswahl von 5 Bildern mit ihren Bounding Boxen und Labels.
        """
        # Sicherstellen, dass wir nicht mehr Bilder anzeigen als wir haben
        num_images = min(num_images, len(self.data))
        
        # Zufällige Indizes für die Bilder auswählen
        random_indices = random.sample(range(len(self.data)), num_images)
        
        # Subplot für die Anzeige der Bilder
        _, axes = plt.subplots(1, num_images, figsize=(15, 15))
        if num_images == 1:
            axes = [axes]  # Falls nur ein Bild, sicherstellen, dass axes eine Liste ist

        for ax, idx in zip(axes, random_indices):
            # Bild, Bounding Box und Label holen
            image, label, bbox = self[idx]
            
            # Umwandeln in numpy-Array, um es mit Matplotlib darzustellen
            image_np = image.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
            
            # Bild anzeigen
            ax.imshow(image_np, cmap='gray')
            
            # Wenn eine Bounding Box vorhanden ist, zeichne sie
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           edgecolor='r', facecolor='none', linewidth=2))

            # Label anzeigen
            ax.set_title(f"Label: {label}")
            ax.axis("off")  # Achsen ausblenden

        plt.tight_layout()
        plt.show()


from torch.utils.data import DataLoader

import math

def visualize_dataloader_batch_with_bboxes(dataloader):
    """
    Visualisiert den ersten Batch aus einem DataLoader mit Bildern, Bounding Boxen und Labels,
    aufgeteilt in zwei Reihen.
    
    Args:
        dataloader (DataLoader): Ein DataLoader-Objekt, das Batches des Datasets liefert.
    """
    # Den ersten Batch aus dem DataLoader holen
    for batch in dataloader:
        images,labels, bboxes = batch  # Batch-Elemente entpacken
        num_images = len(images)  # Anzahl der Bilder im Batch

        # Berechnung der Anzahl der Spalten und Reihen für zwei Reihen
        num_rows = 2
        num_cols = math.ceil(num_images / num_rows)

        # Subplot für die Anzeige der Bilder mit zwei Reihen
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axes = axes.flatten()  # Umwandeln in eine flache Liste für einfachen Zugriff

        for ax, image, label, bbox in zip(axes, images, labels, bboxes):
            # Bild in numpy-Array umwandeln, um es mit Matplotlib darzustellen
            image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

            # Bild anzeigen
            ax.imshow(image_np, cmap='gray')

            # Bounding Box anzeigen, falls vorhanden
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox.numpy()
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           edgecolor='r', facecolor='none', linewidth=2))

            # Label anzeigen
            ax.set_title(f"Label: {label}")
            ax.axis("off")  # Achsen ausblenden

        # Falls es weniger Bilder als Subplots gibt, die leeren Achsen ausblenden
        for i in range(num_images, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
        break  # Nur den ersten Batch anzeigen



