import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_sample_input(dataset_directory):
    TARGET_SIZE = 224 
    
    data_transform = transforms.Compose([
        # FIX: Resize to 256x256 and REMOVE CenterCrop(224) 
        # to ensure the input to the model is 256x256.
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    # Point ImageFolder to the root, which contains the class folders (OTHERS, XRAY)
    full_dataset = datasets.ImageFolder(dataset_directory, data_transform)
    
    # Use shuffle=False for reproducible ONNX export
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    sample_input_tensor, _ = next(iter(dataloader))
    
    return sample_input_tensor, num_classes
