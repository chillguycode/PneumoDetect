import os
from os.path import exists
import torch
import torchvision.models as models
# Assuming loading_sample_input.py provides the correct function and input format
import loading_sample_input as loading 
from pathlib import Path
import sys

from cnn import CNN 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
# FIX: Corrected a potential typo in the file name
PTH_MODEL_PATH = SAVED_MODELS_DIR / "cnn_pneumodetect_best.pth" 
DATASET_PATH = PROJECT_ROOT / "dataset/guard_model_dataset"
ONNX_FILE_NAME = SAVED_MODELS_DIR / "cnn_guard_best.onnx"
MODEL_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print("--- Starting the export ---")

    # Ensuring the directory
    SAVED_MODELS_DIR.mkdir(exist_ok=True)

    # The function get_sample_input must return a (batch_size, channels, H, W) tensor
    # and the number of classes (num_classes).
    sample_input, num_classes = loading.get_sample_input(DATASET_PATH)
    print("Sample Input Loaded")

    print("Loading model architecture")
    
    # 1. Load the state dictionary first
    model_state_dict = torch.load(PTH_MODEL_PATH, map_location=MODEL_DEVICE)
    
    # 2. Instantiate the custom CNN model correctly
    # Your CNN constructor requires 'in_channels' (which is 3 for RGB) and 'num_classes'.
    model = CNN(in_channels=3, num_classes=num_classes)
    
    # 3. CRITICAL FIX: The logic below is for pre-trained models. Your custom CNN 
    # already defines its final fc1 layer correctly, so we remove the unnecessary 
    # modification steps.
    # The loaded model state dictionary should match the newly initialized model.
    # final_layer = cast(torch.nn.Linear, model.classifier[1]) # REMOVE
    # num_ftrs = final_layer.in_features # REMOVE
    # model.classifier[1] = torch.nn.Linear(num_ftrs,num_classes) # REMOVE
    
    # 4. Load the weights
    model.load_state_dict(model_state_dict)
    
    # 5. Set model to evaluation mode
    model.eval() 
    
    # 6. Move model and sample input to the correct device
    model.to(MODEL_DEVICE)
    sample_input = sample_input.to(MODEL_DEVICE)
    
    print("Model Loaded and set to evaluation mode")

    # --- ONNX Export ---
    print("\n--- Starting ONNX Export ---")
    torch.onnx.export(
        model,
        sample_input,
        ONNX_FILE_NAME,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input':{0: 'batch_size'},
                      'output':{0: 'batch_size'}}
    )

    print("\nModel Exported Successfully!")
    print(f"Model Exported To: {ONNX_FILE_NAME}")
