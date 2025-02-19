import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.metrics import confusion_matrix


class CapsuleNetwork(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels_conv=1,
            out_channels_conv=256,
            kernel_size_conv=9,
            stride_conv=1,
            padding_conv=0,
            num_capsules_prime=8,
            out_channels_prime=32,
            kernel_size_prime=9,
            stride_prime=2,
            padding_prime=0,
            num_classes=2,
            out_channels_digit=16,
            hidden_dim=512,
            train_on_gpu=True):

        # initialize CapsNet
        super(CapsuleNetwork, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_on_gpu = train_on_gpu

        # Create initial Convolution Layer
        self.conv_layer = ConvLayer(
            in_channels_conv, out_channels_conv, kernel_size_conv, stride_conv, padding_conv)
        
        # Size of the image after passing the first conv layer (width/height)
        cropped_size = math.floor(
            ((image_size - kernel_size_conv + 2 * padding_conv) / stride_conv) + 1)
        
        # Initialze Primary Caps Layer 
        self.primary_capsules = PrimaryCaps(
            num_capsules_prime,
            in_channels=out_channels_conv,
            out_channels=out_channels_prime,
            kernel_size=kernel_size_prime,
            stride=stride_prime,
            padding=padding_prime)
        
        # Size of the image after passing the primary capsule (width/height)
        cropped_size = math.floor(
            ((cropped_size - kernel_size_prime + 2 * padding_prime) / stride_prime) + 1)
        
        # Initialize Digit Caps Layer
        self.digit_capsules = DigitCaps(
            num_classes=num_classes,
            in_channels=num_capsules_prime,
            out_channels=out_channels_digit,
            previous_out_channels=out_channels_prime,
            cropped_size=cropped_size,
            train_on_gpu=train_on_gpu)
        
        # Initialize Decoder (for image reconstruction)
        self.decoder = Decoder(image_size=image_size, input_vector_length=out_channels_digit,
                               hidden_dim=hidden_dim, num_classes=num_classes, train_on_gpu=train_on_gpu)

        # Define Pipeline
    def forward(self, images):
        # Conv & PrimeCaps layer in one
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        # DigitCaps Layer
        caps_output = self.digit_capsules(
            primary_caps_output).squeeze().transpose(0, 1)
        # Decoder/Reconsturction Layer
        reconstructions, y = self.decoder(caps_output)
        return caps_output, reconstructions, y
        
        # Training
    def train_model(self, train_loader, val_loader, criterion, optimizer, n_epochs, early_stopping, print_every=1):
        # Initialize list of train losses
        losses = []
        self.train_losses = [] 
        self.val_losses = [] 
        
        # Initialize min_loss for best model saving
        min_val_loss = None
        # path to save best model
        weight_file_name = "CAPS_CROPPED/weights/caps_crop_new.pt"

        # Training loop over n_epochs
        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0  
            total_correct = 0  
            total = 0  

            # Iterate over batches in the training loader
            for batch_i, (images, target) in enumerate(train_loader):
                batch_size = images.size(0)
                # Convert target to one-hot encoding
                target = torch.eye(self.num_classes).index_select(
                    dim=0, index=target)
                
                # Move data to GPU if GPU available
                if self.train_on_gpu:
                    images, target = images.cuda(), target.cuda()

                # Compute loss and update weights
                optimizer.zero_grad()
                caps_output, reconstructions, y = self.forward(images)
                loss = criterion(caps_output, target, images, reconstructions)

                loss.backward()  
                optimizer.step()  

                train_loss += loss.item()  
                _, pred = torch.max(y.data.cpu(), 1)
                _, target_shape = torch.max(target.data.cpu(), 1)

                total += target.size(0)  
                total_correct += pred.eq(target_shape).sum().item()


            # Compute average loss per epoch
            avg_train_loss = train_loss / len(train_loader)  
            losses.append(avg_train_loss)
            self.train_losses.append(avg_train_loss)

            # Compute training accuracy
            train_accuracy = 100 * total_correct / total

            # ---- Validation ----
            
            # Initialize list of val losses
            val_running_loss = 0.0  
            val_correct = 0  
            val_total = 0 

            # No gradient calculation during validation
            with torch.no_grad():  
                for batch_i, (images, labels) in enumerate(val_loader):

                    # Convert labels to one-hot encoding
                    target = torch.eye(self.num_classes).index_select(
                        dim=0, index=labels)

                    # Move data to GPU if available
                    if self.train_on_gpu:
                        images, target = images.cuda(), target.cuda()
                        caps_output, reconstructions, y = self.forward(images)

                        # Compute loss
                        val_loss = criterion(caps_output, target, images, reconstructions)
                        val_running_loss += val_loss.item()  

                    # Predictions and accuracy calculation
                        _, val_predicted = torch.max(y.data.cpu(), 1)
                        _, target_shape = torch.max(target.data.cpu(), 1)
                        val_correct += val_predicted.eq(target_shape).sum().item()
                        val_total += target.size(0)

                # Compute average validation loss and accuracy
                val_epoch_loss = val_running_loss / len(val_loader.dataset)
                val_epoch_acc = 100 * val_correct / val_total
                self.val_losses.append(val_epoch_loss)

                # Print results for the epoch
                print(f"Epoch {epoch}/{n_epochs} ----- "
                      f"Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}% ----- "
                      f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

                # Save best model if validation loss improves
                if min_val_loss is None or val_epoch_loss < min_val_loss:
                    min_val_loss = val_epoch_loss
                    checkpoint = {
                        'model_state_dict': self.state_dict(),
                        'loss': val_epoch_loss
                    }
                    os.makedirs(os.path.dirname(
                        weight_file_name), exist_ok=True)
                    torch.save(checkpoint, weight_file_name)
                    print("New best model saved")

                # Check early stopping condition
                early_stopping(val_epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        # Return collected losses
        return losses, self.val_losses 

    # Confusion Matrix
    def display_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))

        threshold = cm.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i,j]}", horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    # Testing
    def test_model(self, criterion, test_loader):
        class_correct = list(0. for i in range(self.num_classes)) # Correct predictions per class
        class_total = list(0. for i in range(self.num_classes)) # Total samples per class
        true_labels = []
        pred_labels = []
        test_loss = 0
        self.eval() # Set model to evaluation mode

        for batch_i, (images, target) in enumerate(test_loader):
            # Convert labels to one-hot encoding          
            target = torch.eye(self.num_classes).index_select(dim=0, index=target)
            batch_size = images.size(0)

            # Move data to GPU if available
            if self.train_on_gpu:
                images, target = images.cuda(), target.cuda()
            
            caps_output, reconstructions, y = self.forward(images)
           
            # Compute loss
            loss = criterion(caps_output, target, images, reconstructions)
            test_loss += loss.item()

            # Get predictions
            _, pred = torch.max(y.data.cpu(), 1)
            _, target_shape = torch.max(target.data.cpu(), 1)

            # Check correctness
            correct = np.squeeze(pred.eq(target_shape.data.view_as(pred)))

            # Store true and predicted labels for confusion matrix
            true_labels.extend(target_shape.tolist())
            pred_labels.extend(pred.tolist())

            # Update per-class accuracy
            for i in range(batch_size):
                label = target_shape.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # Compute average test loss
        avg_test_loss = test_loss / len(test_loader)
        print('Test Loss: {:.8f}\n'.format(avg_test_loss))

        # Print per-class accuracy
        for i in range(self.num_classes):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %.2f%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' %
                      (class_total[i]))
        # Print overall test accuracy
        print('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

        # Display confusion matrix        
        self.display_confusion_matrix(torch.tensor(true_labels), torch.tensor(pred_labels))

        # return last batch of capsule vectors, images, reconstructions
        return caps_output, images, reconstructions

    def run_model(self, test_loader):
        # initialize empty lists for predictions and their image paths
        all_preds = []
        all_paths = []

        # Disable gradient calculation 
        with torch.no_grad():
            for batch_i, (images, paths) in enumerate(test_loader):
                
                # Move images to GPU if available
                if self.train_on_gpu:
                    images = images.cuda()

                # Forward pass through the model
                caps_output, reconstructions, y = self.forward(images)

                # Get predicted class labels
                _, pred = torch.max(y.data.cpu(), 1)
                
                # Store predictions and file paths
                all_preds.extend(pred.tolist())
                all_paths.extend(list(paths))
                
                print("Batch number: ", batch_i)

        # Return predictions and corresponding file paths
        return all_preds, all_paths

# Define Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        # Define a 2D convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # Apply convolution followed by ReLU activation
        features = F.relu(self.conv(x))
        return features



class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride, padding):
        super(PrimaryCaps, self).__init__()
        self.out_channels = out_channels
        # Create multiple convolutional layers, one for each capsule
        self.capsules = nn.ModuleList([nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding) for _ in range(num_capsules)])

    def forward(self, x):
        batch_size = x.size(0)
        # Apply each capsule-specific convolution, reshape output, and concatenate
        u = [capsule(x).view(batch_size, self.out_channels * capsule(x).shape[2]
                             * capsule(x).shape[2], 1) for capsule in self.capsules]
        
        # Concatenate capsule outputs
        u = torch.cat(u, dim=-1)

        # Apply non-linear squash function
        u_squash = self.squash(u) 
        return u_squash

    def squash(self, input_tensor):
        # Squash function to normalize capsule outputs
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1+squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, previous_out_channels, cropped_size, train_on_gpu):

        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_classes
        self.in_channels = in_channels  # previous layer's number of capsules
        self.out_channels = out_channels
        self.train_on_gpu = train_on_gpu

        # Randomly initialize weight matrix W to transform capsules from PrimaryCaps to DigitCaps
        self.W = nn.Parameter(torch.randn((self.num_capsules, previous_out_channels*cropped_size*cropped_size,
                                           self.in_channels, self.out_channels)))

    def forward(self, u):
        # Add batch dimension and prepare the input for matrix multiplication
        u = u[None, :, :, None, :]
        
        # Define 4D weight matrix W for transformation
        W = self.W[:, None, :, :, :]

        # Compute u_hat = W * u (transform input vectors using weight matrix)
        u_hat = torch.matmul(u, W)

        # Initialize b_ij (coupling coefficients) to zero initially
        b_ij = torch.zeros(*u_hat.size())
        # moving b_ij to GPU, if available
        if self.train_on_gpu:
            b_ij = b_ij.cuda()

        # Perform dynamic routing and compute the final output capsule vectors v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j  # return final vector outputs

    def squash(self, input_tensor):
        # Squash function to normalize capsule outputs
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1+squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor



class Decoder(nn.Module):
    def __init__(self, image_size, input_vector_length, hidden_dim, num_classes, train_on_gpu):
        super(Decoder, self).__init__()

        # Calculate input dimension based on vector length and number of classes
        input_dim = input_vector_length * num_classes

        # Define the decoder network with 3 fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, image_size ** 2),
            nn.Sigmoid() # Sigmoid for pixel value normalization
        )
        self.num_classes = num_classes
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        # Calculate the class probabilities from the input (capsule vectors)
        classes = (x ** 2).sum(dim=-1)**0.5 # Calculate the length of capsule vectors
        classes = F.softmax(classes, dim=-1) # Apply softmax to get class probabilities

        # select most probable class  for each image in batch 
        _, max_length_indices = classes.max(dim=1)

        # Create a one-hot encoding of the predicted class for each sample
        sparse_matrix = torch.eye(self.num_classes)
        
        # if GPU available pass to GPU
        if self.train_on_gpu:
            sparse_matrix = sparse_matrix.cuda()

        # One-Hot-Encoding of the classes
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)

        # Multiply input with the one-hot encoded class vector
        x = x * y[:, :, None] 

        # Flatten the input before passing it through the decoder
        flattened_x = x.contiguous().view(x.size(0), -1)

        # Decode the flattened vector back to the image space
        reconstructions = self.linear_layers(flattened_x)

        # Return the reconstructed image and the one-hot encoded class labels
        return reconstructions, y


class CapsuleLoss(nn.Module):
    def __init__(self, learning_rate=5e-04):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="sum")  # Mean Squared Error loss for image reconstruction
        self.learning_rate = learning_rate # Learning rate for the loss calculation

    def forward(self, x, labels, images, reconstructions):
        
        batch_size = x.size(0) # Number of images in the current batch
         # Calculate the magnitude of the capsule output vectors (v_c)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True)) # Sum the squares of each vector and take the square root to get the magnitude


        # Compute the left and right part of the margin loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)  # Relu applied to (0.9 - v_c), encourages capsule vectors to be larger than 0.9
        right = F.relu(v_c - 0.1).view(batch_size, -1) # Relu applied to (v_c - 0.1), encourages capsule vectors to be smaller than 0.1 for non-target classes
        
        # Calculate the margin loss: 
        # - When the label is 1 (positive class), the loss encourages the capsule vector magnitude to be above 0.9
        # - When the label is 0 (negative class), the loss encourages the capsule vector magnitude to be below 0.1
        margin_loss = labels * left + 0.5 * (1. - labels) * right

        # Sum the margin loss over all capsules for each image
        margin_loss = margin_loss.sum() # This sums the loss for all capsules and images in the batch

        # Flatten the original images into a 2D tensor (batch_size, image_size^2)
        images = images.view(reconstructions.size()[0], -1)

        # check mean squarred error between reconstruction and original image (reduced as sum)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # The final loss is a weighted sum of the margin loss and the reconstruction loss
        # The reconstruction loss is scaled by the learning rate to control its impact on the final loss
        # Normalize by dividing by the batch size to get an average loss per example
        return (margin_loss + self.learning_rate * reconstruction_loss) / images.size(0)


def softmax(input_tensor, dim=1):
    # Transpose the tensor so that the specified dimension becomes the last one    
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    # transposed.contiguous just checks if the tensor is contiguous in the memory
    # the view means that we reshape the tensor to size (x, last dim of the tensor)
    # in this case the previous 4 dims get multiplied and we keep the 1152 --> (3200, 1152)
    # then the softmax function will be applied to the last dimension (1152)
    softmaxed_output = F.softmax( transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # resize the softmaxed output back to the original dimensions (10, 20, 16, 1, 1152)
    softmaxed_output = softmaxed_output.view(*transposed_input.size())
    # switch back the dimensions --> (10, 20, 1152, 1, 16)
    return softmaxed_output.transpose(dim, len(input_tensor.size()) - 1)


def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):

    # Iterate through the dynamic routing process for the specified number of iterations
    for iteration in range(routing_iterations):
        # Calculate the coupling coefficients c_ij using the softmax function on b_ij
        c_ij = softmax(b_ij, dim=2)

        # Multiply c_ij (coupling coefficients) and u_hat (the transformed input capsules) and sum over the capsules of the previous layer (dim=2)
        # This gives the weighted sum of the input capsules for each output capsule
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        # Apply the squash function to the summed values, which normalizes them to a magnitude between 0 and 1
        v_j = squash(s_j)

        # if not last iteration  continue to update the coupling coefficients b_ij
        if iteration < routing_iterations - 1:
            # Calculate the attention coefficients a_ij by multiplying u_hat with the output capsule v_j
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            # Update b_ij by adding the calculated attention coefficients
            b_ij = b_ij + a_ij

    # Return the final output capsule vector v_j after all iterations
    return v_j
