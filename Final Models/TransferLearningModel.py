import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, Inception_V3_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights

from CNN3D import CNN3D, Custom3DTransform


class TransferLearningModel(nn.Module):
    def __init__(self, classes, model_name, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=0.001):
        super(TransferLearningModel, self).__init__()
        self.classes = classes
        self.device = device
        self.model_name = model_name
        self.model = None
        # Initialize the model based on model_name
        if self.model_name == "resnet":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, len(self.classes))        
        elif self.model_name == "densenet":
            self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, len(self.classes))
        elif self.model_name == "inception":
            self.model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, len(self.classes))
        elif self.model_name == "efficientnet":
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, len(self.classes))
        elif self.model_name == "vit":
            self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, len(self.classes))
        elif self.model_name == "3dcnn":
            self.get_transforms()
            self.model = CNN3D(image_size=self.resize_dim[0], classes=self.classes)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")
        
        # Set the transforms based on model requirements
        self.train_transforms, self.test_transforms = self.get_transforms()

        # Freeze the feature extractor layers
        for name, param in self.model.named_parameters():
            if "fc" not in name and "classifier" not in name and "heads.head" not in name:  # Leave last layers unfrozen
                param.requires_grad = False
        # Move model to the specified device
        self.model = self.model.to(self.device)

        # Define loss function and learning rate scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5)

    def get_transforms(self):
        # Define common normalization
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.model_name == "inception":
            self.resize_dim = (299, 299)
            # Inception requires 299x299 input images
            train_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(self.resize_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            test_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(self.resize_dim),
                transforms.ToTensor(),
                normalize
            ])
        elif self.model_name == "3dcnn":
            self.resize_dim = (224, 224)
            train_transforms = Custom3DTransform(resize=self.resize_dim, flip_prob=0.5)
            test_transforms = Custom3DTransform(resize=self.resize_dim, flip_prob=0.0)
        else:
            self.resize_dim = (224, 224)
            # Default input size for most other models is 224x224
            train_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(self.resize_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            test_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(self.resize_dim),
                transforms.ToTensor(),
                normalize
            ])

        return train_transforms, test_transforms

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
            if self.model_name == "inception":
                self.model.aux_logits = True
            running_loss = 0.0
            correct = 0
            total = 0         

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Handle Inception model output
                if self.model_name == "inception":
                    logits = outputs.logits  # Use the main logits output
                    aux_logits = outputs.aux_logits
                    loss = self.criterion(logits, labels) + 0.4 * self.criterion(aux_logits, labels)  # Combine with auxiliary loss
                else:
                    logits = outputs
                    loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)

            self.model.eval()
            if self.model_name == "inception":
                self.model.aux_logits = False
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