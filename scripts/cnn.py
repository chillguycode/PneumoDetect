import torch
import torch.nn.functional as F
import torch.nn as nn



class CNN(nn.Module):
    """
    A 3-layer Convolutional Neural Network (CNN) with 2 Fully Connected (FC) 
    layers for binary classification.
    """
    def __init__(self, in_channels, num_classes=2):
        super(CNN, self).__init__()
        
        # NOTE: Input image size is 224x224 based on data transforms.
        
        # --- 1. Convolutional Layer 1: in_channels -> 128 (Output size after pool: 112x112) ---
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # --- 2. Convolutional Layer 2: 128 -> 64 (Output size after pool: 56x56) ---
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- 3. Convolutional Layer 3: 64 -> 32 (Output size after pool: 28x28) ---
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # --- Fully Connected Layers (2 Layers) ---
        # The input size is (Final Channels) * (Final Height) * (Final Width)
        # 32 channels * 28 height * 28 width = 25088 (Correct for 224x224 input)
        FC1_INPUT_SIZE = 32 * 28 * 28 # FIXED: Now 25088 features
        FC2_HIDDEN_SIZE = 512           

        # FC Layer 1 (Hidden Layer)
        self.fc1 = nn.Linear(FC1_INPUT_SIZE, FC2_HIDDEN_SIZE)
        
        # FC Layer 2 (Output Layer - Logits)
        self.fc2 = nn.Linear(FC2_HIDDEN_SIZE, num_classes)


    def forward(self, x):
        # --- 3 Convolutional Blocks ---
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # --- Flattening ---
        x = x.reshape(x.shape[0], -1) 

        # --- 2 Fully Connected Layers ---
        # 1. Hidden Layer with ReLU activation
        x = F.relu(self.fc1(x)) 
        
        # 2. Output Layer (Logits)
        x = self.fc2(x) 
        
        return x
