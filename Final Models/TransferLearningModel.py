import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models


class TransferLearningModel(nn.Module):
    def __init__(self, classes, model_name, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=0.001):
        super(TransferLearningModel, self).__init__()
        self.classes = classes
        self.device = device
        self.model_name = model_name
        self.model = None
        if self.model_name == "resnet":
            self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=2)

    def train(self, train_loader, val_loader, early_stopping, epochs=5):
        min_val_loss = None
        weight_file_name = f"weights/{self.model_name}.pt"
        self.train_losses = []
        self.val_losses = []

        if min_val_loss is None and os.path.exists(weight_file_name):
            print("Weights already exist, start from best previous values.")
            checkpoint = torch.load(weight_file_name, weights_only=False)
            min_val_loss = checkpoint["loss"]
            self.model.load_state_dict(checkpoint["model_state_dict"])

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0         

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)

            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    val_loss = self.criterion(outputs, labels)
                    val_running_loss += val_loss.item() * inputs.size(0)
                    _, val_predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += val_predicted.eq(labels).sum().item()

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_acc = 100 * val_correct / val_total
            self.val_losses.append(val_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} ----- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% ----- Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

            if min_val_loss == None or val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'loss': val_epoch_loss
                }
                torch.save(checkpoint, weight_file_name)
            self.scheduler.step(val_epoch_loss)
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
        
        avg_loss = running_loss / len(test_loader.dataset)
        accuracy = 100 * correct / total
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        print(f"Evaluation Accuracy on unseen data: {accuracy}")
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Reds, values_format='d')
        plt.title("Confusion Matrix")
        plt.show()
    
    def predict(self, inputs):
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()
    
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()