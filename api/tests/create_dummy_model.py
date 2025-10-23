import torch
import torch.nn as nn
from pathlib import Path


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,2)
    
    def forward(self,x):
        return self.linear(x.view(1,-1)[:,0].unsqueeze(0))

def generate_dummy_model(output_path: Path):
    if output_path.exists():
        print(f"Dummy model already exists at {output_path}")
        return

    print(f"Generating dummy ONNX model at {output_path}")
    model = DummyModel()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            )
    print("Dummmy model generated successfully")

if __name__ == "__main__":
    MODEL_DIR = Path(__file__).parent.parent.parent/ "saved_models_test"
    generate_dummy_model(MODEL_DIR/"dummy_model.onnx")
