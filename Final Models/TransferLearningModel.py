import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models


class TransferLearningModel:
    def __init__(self, classes, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=0.001):
        self.classes = classes
        self.device = device
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)

    def train(self, train_loader, early_stopping, epochs=5):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            print(f"Epoch {epoch+1}/{epochs}")

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
            print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            early_stopping(epoch_loss)
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
            
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Reds, values_format='d')
        plt.title("Confusion Matrix")
        plt.show()
        return accuracy
    
    def predict(self, inputs):
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()