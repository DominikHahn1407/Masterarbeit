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
        super(CapsuleNetwork, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_on_gpu = train_on_gpu
        self.conv_layer = ConvLayer(in_channels_conv, out_channels_conv, kernel_size_conv, stride_conv, padding_conv)        
        # Size of the image after passing the first conv layer (width/height)
        cropped_size = math.floor(((image_size - kernel_size_conv + 2 * padding_conv) / stride_conv) + 1)
        self.primary_capsules = PrimaryCaps(
            num_capsules_prime, 
            in_channels=out_channels_conv, 
            out_channels=out_channels_prime, 
            kernel_size=kernel_size_prime, 
            stride=stride_prime, 
            padding=padding_prime)
        # Size of the image after passing the primary capsule (width/height)
        cropped_size = math.floor(((cropped_size - kernel_size_prime + 2 * padding_prime) / stride_prime) + 1)
        self.digit_capsules = DigitCaps(
            num_classes=num_classes, 
            in_channels=num_capsules_prime, 
            out_channels=out_channels_digit, 
            previous_out_channels=out_channels_prime, 
            cropped_size=cropped_size, 
            train_on_gpu=train_on_gpu)
        self.decoder = Decoder(image_size=image_size, input_vector_length=out_channels_digit, hidden_dim=hidden_dim, num_classes=num_classes, train_on_gpu=train_on_gpu)

    def forward(self, images):
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        # squeeze removes dimensions of size 1 and then transpose 0 and 1 dim (10,20,1,1,16)-->(20,10,16)
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0,1)
        reconstructions, y = self.decoder(caps_output)
        return caps_output, reconstructions, y
    def train_model(self, train_loader, val_loader, criterion, optimizer, n_epochs, early_stopping, print_every=1):
        # Liste zum Speichern der Verluste während des Trainings
        losses = []
        self.train_losses = []  # Liste für die Trainingsverluste pro Epoche
        self.val_losses = []  # Liste für die Validierungsverluste pro Epoche
        min_val_loss = None  # Variable für den minimalen Validierungsverlust, um die besten Gewichte zu speichern
        weight_file_name = "CAPS_CROPPED/weights/caps_crop_new.pt"  # Der Pfad, unter dem die Modellgewichte gespeichert werden

        # Training über n_epochs Epochen
        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0  # Akkumulierte Trainingsverlust der aktuellen Epoche
            total_correct = 0  # Anzahl der korrekt klassifizierten Bilder
            total = 0  # Gesamtzahl der Bilder in der aktuellen Epoche
            

            # Schleife über alle Batches im Trainings-Loader
            for batch_i, (images, target) in enumerate(train_loader):
                batch_size = images.size(0)
                # Ziel (target) in One-Hot-Encoding umwandeln
                target = torch.eye(self.num_classes).index_select(dim=0, index=target)
                if self.train_on_gpu:
                    # Falls Training auf der GPU, dann Bilder und Targets auf die GPU verschieben
                    images, target = images.cuda(), target.cuda()

                optimizer.zero_grad()  # Setzt den Gradienten des Optimizers auf Null
                # Vorwärtsdurchlauf durch das Modell
                caps_output, reconstructions, y = self.forward(images)
                
                # Berechnung des Verlusts
                loss = criterion(caps_output, target, images, reconstructions)
                loss.backward()  # Berechnung der Gradienten
                optimizer.step()  # Schritt des Optimizers, um die Gewichte zu aktualisieren

                train_loss += loss.item()  # Akkumulieren des Verlusts
                _, pred = torch.max(y.data.cpu(), 1)
                _, target_shape = torch.max(target.data.cpu(), 1)
                # correct += np.squeeze(pred.eq(target_shape.data.view_as(pred)))
                total += target.size(0)  # Anzahl der Bilder in diesem Batch
                # print(f"total: {total}")
               
                total_correct += pred.eq(target_shape).sum().item()  # Anzahl der korrekt klassifizierten Beispiele
                # print(f"correct pred {total_correct}")
            # Sobald das angegebene print_every Intervall erreicht wird, den Durchschnittsverlust berechnen
            avg_train_loss = train_loss / len(train_loader)  # Durchschnitt pro Epoche
            losses.append(avg_train_loss)  # Speichern des Durchschnittsverlusts
            self.train_losses.append(avg_train_loss)  # Speichern des Durchschnittsverlusts für späteres Plotten
                
            train_accuracy = 100 * total_correct / total

            # Modell in den Evaluierungsmodus setzen für die Validierung
            
            val_running_loss = 0.0  # Akkumulierter Verlust während der Validierung
            val_correct = 0  # Anzahl der korrekt klassifizierten Validierungsbilder
            val_total = 0  # Gesamtanzahl der Validierungsbilder

            with torch.no_grad():  # Keine Gradientenberechnung während der Validierung
                for batch_i, (images, labels) in enumerate(val_loader):
                                
                    # Labels in One-Hot-Encoding umwandeln
                    target = torch.eye(self.num_classes).index_select(dim=0, index=labels)
                    if self.train_on_gpu:
                        images, target = images.cuda(), target.cuda()  # Bilder und Targets auf GPU verschieben
                        # Vorwärtsdurchlauf durch das Capsule Network
                        caps_output, reconstructions, y = self.forward(images)
                        # Verlustberechnung
                        val_loss = criterion(caps_output, target, images, reconstructions)
                        val_running_loss += val_loss.item()  # Verlust akkumulieren

                        # Vorhersagen und Genauigkeit berechnen
                        _, val_predicted = torch.max(y.data.cpu(), 1)
                        _, target_shape = torch.max(target.data.cpu(), 1)

                        val_correct += val_predicted.eq(target_shape).sum().item()  # Richtig klassifizierte Beispiele
                        val_total += target.size(0)

                # Durchschnittlicher Validierungsverlust und Genauigkeit berechnen
                val_epoch_loss = val_running_loss / len(val_loader.dataset)
                val_epoch_acc = 100 * val_correct / val_total
                self.val_losses.append(val_epoch_loss)  # Speichern des Verlusts in der Liste

                # Ausgabe der Ergebnisse für diese Epoche
                print(f"Epoch {epoch}/{n_epochs} ----- "
                    f"Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}% ----- "
                    f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

                # Wenn der aktuelle Validierungsverlust besser ist als der bisher beste, speichere das Modell
                # Speichern der besten Modellgewichte basierend auf dem minimalen Validierungsverlust
                if min_val_loss is None or val_epoch_loss < min_val_loss:
                    min_val_loss = val_epoch_loss
                    checkpoint = {
                        'model_state_dict': self.state_dict(),
                        'loss': val_epoch_loss
                    }
                    os.makedirs(os.path.dirname(weight_file_name), exist_ok=True)
                    torch.save(checkpoint, weight_file_name)
                    print("New best model saved")

                early_stopping(val_epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        

        # # Verlust-Diagramm speichern
        # loss_plot_path = weight_file_name.replace('.pt', '_loss_plot.png')
        # self.plot_loss(save_path=loss_plot_path)  # Plot-Funktion für den Verlust

        return losses, self.val_losses  # Rückgabe der gesammelten Verluste


    # def train_model(self, train_loader, val_loader, criterion, optimizer, n_epochs, early_stopping, print_every=1):
    #     losses = []
    #     self.train_losses = []
    #     self.val_losses = []
    #     min_val_loss=None
    #     weight_file_name = "weights/caps/caps_crop.pt"
    #     for epoch in range(1, n_epochs+1):
    #         train_loss = 0.0
    #         correct = 0
    #         total = 0 
    #         self.train()
    #         for batch_i, (images, target) in enumerate(train_loader):
    #             target = torch.eye(self.num_classes).index_select(dim=0, index=target)
    #             if self.train_on_gpu:
    #                 images, target = images.cuda(), target.cuda()
    #             optimizer.zero_grad()
    #             caps_output, reconstructions, y = self.forward(images)
    #             loss = criterion(caps_output, target, images, reconstructions)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item()

    #             total += target.size(0)
    #             correct += caps_output.eq(target).sum().item()
    #             epoch_acc = 100 * correct / total
    #             self.train_losses.append(avg_train_loss)
            

    #             if batch_i != 0 and batch_i % print_every == 0:
    #                 avg_train_loss = train_loss/print_every
    #                 losses.append(avg_train_loss)
    #                 train_loss = 0

    #         epoch_acc = 100 * correct / total

    #         self.model.eval()

    #         val_running_loss = 0.0
    #         val_correct = 0
    #         val_total = 0

    #         with torch.no_grad():
    #             for inputs, labels in val_loader:
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)
    #                 outputs = self.model(inputs)
    #                 val_loss = self.criterion(outputs, labels)
    #                 val_running_loss += val_loss.item() * inputs.size(0)
    #                 _, val_predicted = outputs.max(1)
    #                 val_total += labels.size(0)
    #                 val_correct += val_predicted.eq(labels).sum().item()

    #         val_epoch_loss = val_running_loss / len(val_loader.dataset)
    #         val_epoch_acc = 100 * val_correct / val_total
    #         self.val_losses.append(val_epoch_loss)
    #         print(f"Epoch {epoch+1}/{n_epochs+1} ----- Loss: {avg_train_loss:.4f}, Accuracy: {epoch_acc:.2f}% ----- Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

    #         if min_val_loss == None or val_epoch_loss < min_val_loss:
    #             min_val_loss = val_epoch_loss
    #             checkpoint = {
    #                 'model_state_dict': self.model.state_dict(),
    #                 'loss': val_epoch_loss
    #             }
    #             torch.save(checkpoint, weight_file_name)
    #         self.scheduler.step(val_epoch_loss)
    #         early_stopping(val_epoch_loss)
    #         if early_stopping.early_stop:
    #             print("Early stopping triggered")
    #             break
    #     loss_plot_path = weight_file_name.replace('.pt', '_loss_plot.png')
    #     self.plot_loss(save_path=loss_plot_path)   

    #                 # print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch, avg_train_loss))
    #              # reset accumulated training loss
    #     return losses
    
    def display_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))

        threshold = cm.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i,j]}", horizontalalignment="center", color="white" if cm[i,j] > threshold else "black")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()
    
    def test_model(self, criterion, test_loader):
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        true_labels = []
        pred_labels = []
        test_loss = 0
        self.eval()

        for batch_i, (images, target) in enumerate(test_loader):
            target = torch.eye(self.num_classes).index_select(dim=0, index=target)
            batch_size = images.size(0)
            if self.train_on_gpu:
                images, target = images.cuda(), target.cuda()
            caps_output, reconstructions, y = self.forward(images)
            loss = criterion(caps_output, target, images, reconstructions)
            test_loss += loss.item()
            _, pred = torch.max(y.data.cpu(), 1)
            _, target_shape = torch.max(target.data.cpu(), 1)
            correct = np.squeeze(pred.eq(target_shape.data.view_as(pred)))
            true_labels.extend(target_shape.tolist())
            pred_labels.extend(pred.tolist())

            for i in range(batch_size):
                label = target_shape.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        avg_test_loss = test_loss / len(test_loader)
        print('Test Loss: {:.8f}\n'.format(avg_test_loss))
        for i in range(self.num_classes):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %.2f%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))

        print('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        self.display_confusion_matrix(torch.tensor(true_labels), torch.tensor(pred_labels))
        # return last batch of capsule vectors, images, reconstructions
        return caps_output, images, reconstructions
    
    def run_model(self, test_loader):
        all_preds=[]
        all_paths=[]
        with torch.no_grad():
            for batch_i, (images, paths) in enumerate(test_loader):               
                if self.train_on_gpu:
                    images= images.cuda()
                caps_output, reconstructions, y = self.forward(images)
                
                _, pred = torch.max(y.data.cpu(), 1)
                # Ergebnisse des aktuellen Batches sammeln
                all_preds.extend(pred.tolist())  # Vorhersagen zur Liste hinzufügen
                all_paths.extend(list(paths))   # Dateipfade zur Liste hinzufügen
                print("Batch number: ",batch_i)
                    
        return all_preds, all_paths

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # new_image_size = ((input_size - kernel + 2*padding) / stride) + 1 f.e. ((280 - 9)/1)+1 = 272
        # (batch_size, out_channels, 
        features = F.relu(self.conv(x))
        return features
    
class PrimaryCaps(nn.Module):
    # out channels is in channels divided by number of capsules
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride, padding):
        super(PrimaryCaps, self).__init__()
        self.out_channels = out_channels
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) for _ in range(num_capsules)])

    def forward(self, x):
        batch_size = x.size(0)
        u = [capsule(x).view(batch_size, self.out_channels * capsule(x).shape[2] * capsule(x).shape[2], 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        u_squash = self.squash(u)
        return u_squash
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1+squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor
    
class DigitCaps(nn.Module):    
    def __init__(self, num_classes, in_channels, out_channels, previous_out_channels, cropped_size, train_on_gpu):
        '''Constructs an initial weight matrix, W, and sets class variables.
           param num_capsules: number of capsules to create
           param previous_layer_nodes: dimension of input capsule vector, default value = 1152
           param in_channels: number of capsules in previous layer, default value = 8
           param out_channels: dimensions of output capsule vector, default value = 16
           '''
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_classes
        # self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules
        self.out_channels = out_channels
        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.train_on_gpu = train_on_gpu
        self.W = nn.Parameter(torch.randn((self.num_capsules, previous_out_channels*cropped_size*cropped_size, 
                                    self.in_channels, self.out_channels)))

    def forward(self, u):
        '''Defines the feedforward behavior.
           param u: the input; vectors from the previous PrimaryCaps layer
           return: a set of normalized, capsule output vectors
           '''
        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :] # doppelpunkt nimmt die dimension aus original, none added eine dimension mit länge 1
        # 4D weight matrix
        W = self.W[:, None, :, :, :]
        
        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)
        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size())
        # moving b_ij to GPU, if available
        if self.train_on_gpu:
            b_ij = b_ij.cuda()

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)
        return v_j # return final vector outputs
    
    
    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        # same squash function as before
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor
    
class Decoder(nn.Module):
    def __init__(self, image_size, input_vector_length, hidden_dim, num_classes, train_on_gpu):
        super(Decoder, self).__init__()
        input_dim = input_vector_length * num_classes
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, image_size ** 2),
            nn.Sigmoid()
        )
        self.num_classes = num_classes
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        # x shape (20,10,16)
        # x**2 shape -> (20,10,16)
        # sum(dim-1) --> shape (20,10)
        # classes shape (20,10) --> batch / amount of classes
        classes = (x ** 2).sum(dim=-1)**0.5
        # Applying softmax to the last dimension --> get most probable class at each batch
        classes = F.softmax(classes, dim=-1)      
        # select most probable class  for each image in batch array of length 20
        _, max_length_indices = classes.max(dim=1)
        # create unity matrix of the size of the amount of classes
        sparse_matrix = torch.eye(self.num_classes)
        if self.train_on_gpu:
            sparse_matrix = sparse_matrix.cuda()
        # One-Hot-Encoding of the classes
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        x = x * y[:, :, None] # (20,10,16) * (20,10,1)
        # flat x --> (20,160) 10*16
        flattened_x = x.contiguous().view(x.size(0), -1)
        reconstructions = self.linear_layers(flattened_x)
        # y = one hot encoded labels
        return reconstructions, y
    
class CapsuleLoss(nn.Module):
    def __init__(self, learning_rate=5e-04):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="sum")
        self.learning_rate = learning_rate

    def forward(self, x, labels, images, reconstructions):
        # x shape: 20,10,16, label shape: 20,10, image shape: 20, 1, 28, 28, reconstruction shape: 20, 784
        batch_size = x.size(0)
        # square x + sum letzte dim -> (20,10,1)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        # relu for 0.9 - v_c value at each position of (20,10,1) then resize to (20,10)
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        # relu for v_c value - 0.1 at each position of (20,10,1) then resize to (20,10)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1. - labels) * right # some multiplications
        # sum above (20,10) --> one value
        margin_loss = margin_loss.sum()
        # bring images to same size as reconstructions (flatten)
        images = images.view(reconstructions.size()[0], -1)
        # check mean squarred error between reconstruction and original image (reduced as sum)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        # calculate the loss by adding a learning factor * reconstruction loss to the margin loss and divide it by 
        # the batch size
        return (margin_loss + self.learning_rate * reconstruction_loss) / images.size(0)

    
def softmax(input_tensor, dim=1):
    # switch the dimension dim with the last dimension of the transposed input
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    # transposed.contiguous just checks if the tensor is contiguous in the memory
    # the view means that we reshape the tensor to size (x, last dim of the tensor)
    # in this case the previous 4 dims get multiplied and we keep the 1152 --> (3200, 1152)
    # then the softmax function will be applied to the last dimension (1152)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # resize the softmaxed output back to the original dimensions (10, 20, 16, 1, 1152)
    softmaxed_output = softmaxed_output.view(*transposed_input.size())
    # switch back the dimensions --> (10, 20, 1152, 1, 16)
    return softmaxed_output.transpose(dim, len(input_tensor.size()) - 1)

def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iteration in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        # multiply c_ij and u_hat with both dimension (10,20,1152,1,16) --> results in shape (10,20,1152,1,16)
        # then sum along the last layer nodes dimension 1152 --> (10,20,1,1,16)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        # squash values between 0 and 1 still dim (10,20,1,1,16)
        v_j = squash(s_j)
        # if not last iteration
        if iteration < routing_iterations - 1:
            # (10,20,1152,1,16) * (10,20,1,1,16) --> (10,20,1152,1,16)
            # sum at last dimension --> (10,20,1152,1,1)
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            # add the calculated value to the initially set 0 b_ij
            b_ij = b_ij + a_ij
    return v_j

def display_images(images, reconstructions):
    image_size = images.shape[2]
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.view(-1, 1, image_size, image_size)
    reconstructions = reconstructions.data.cpu().numpy()
    _, axs = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(26,5))
    for images, row in zip([images, reconstructions], axs):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)