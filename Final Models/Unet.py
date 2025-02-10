import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        ''' Initializes the U-Net model.

        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes for classification.
            device (str): Device to run the model on.
        '''
        super(UNet, self).__init__()
        self.flatten_size = None
        self.device = device
        #Encoder (Contracting Path)
        # Creation of 8 Conv Layers with ReLU activasion function to progressively reduce spatial dimension
        self.enc1 = self.double_conv(input_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        #Bottleneck (bridge between the two paths with the highest feature depth)
        self.bottleneck = self.double_conv(512, 1024)
        #Decoder (Expanding Path)
        # Restoration of spatial dimensions by upsampling with Conv Layers and ReLU activation function and a transponation
        self.upconv4 = self.up_conv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.dec3 = self.double_conv(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.dec2 = self.double_conv(256, 128)
        self.upconv1 = self.up_conv(128, 64)
        self.dec1 = self.double_conv(128, 64)
        #Classification
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def double_conv(self, in_channels, out_channels):
        # Function to create two Conv Layers with applied ReLU activasion functions
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        # Function to create a 2D transpose layer
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        #Encoder --> Max Pooling is applied after each 2 Conv Layers
        c1 = self.enc1(x)
        p1 = F.max_pool2d(c1, kernel_size=2)
        c2 = self.enc2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2)
        c3 = self.enc3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2)
        c4 = self.enc4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2)
        #Bottleneck
        c5 = self.bottleneck(p4)
        #Decoder --> here with the catting the contracting path is directly connected to the expanding path
        # at each level of detail
        u4 = self.upconv4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.dec4(u4)
        u3 = self.upconv3(c6)
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.dec3(u3)
        u2 = self.upconv2(c7)
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.dec2(u2)
        u1 = self.upconv1(c8)
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.dec1(u1)
        #Classification --> A flatten layer with a dense layer are inluded for the classification head of the
        # U-Net model
        if self.flatten_size is None:
            self.flatten_size = c9.shape[1] * c9.shape[2] * c9.shape[3]
            self.fc1 = nn.Linear(self.flatten_size, 128).to(self.device)
        flat = c9.view(c9.size(0), -1)
        fc1_out = F.relu(self.fc1(flat))
        output = self.fc2(fc1_out)
        return output