import os
from os.path import exists
import torch
import torchvision.models as models
import loading_sample_input as loading
from pathlib import Path
import sys
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
PTH_MODEL_PATH = SAVED_MODELS_DIR / "best model/efficientnet_v2_s_best.pth"
DATASET_PATH = PROJECT_ROOT / "dataset/chest_xray"
ONNX_FILE_NAME = SAVED_MODELS_DIR / "efficientnet_v2_s_best.onnx"
MODEL_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print("--- Starting the export ---")

    #Ensuring the directory
    SAVED_MODELS_DIR.mkdir(exist_ok=True)

    sample_input, num_classes = loading.get_sample_input(DATASET_PATH)
    print("Sample Input Loaded")

    print("Loading model architecture: EfficientNet V2 Small")
    model_state_dict = torch.load(PTH_MODEL_PATH,map_location=MODEL_DEVICE)
    model = models.efficientnet_v2_s()
    final_layer = cast(torch.nn.Linear, model.classifier[1])
    num_ftrs = final_layer.in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs,num_classes)
    model.load_state_dict(model_state_dict)
    print("Model Loaded")




    onnx_model = torch.onnx.export(
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

    print("Model Exported Successfully")
    print(f"Model Exported To: {ONNX_FILE_NAME}")
