from os import wait
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import torch

class InfPipeline:
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])


        self.input_name = self.session.get_inputs()[0].name
        print(f"Model input name: {self.input_name}")



        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            ])

        self.class_names = ["NORMAL","PNEUMONIA"]
        print("Inference pipeline initialised successfully!")


    def predict(self, image_bytes:bytes) -> dict:

        try:
            image: Image.Image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")


            input_tensor: torch.Tensor = self.transforms(image)


            input_batch = input_tensor.unsqueeze(0)
            onnx_input = {self.input_name: input_batch.numpy()}
            raw_output: list[np.ndarray] = self.session.run(None, onnx_input)
            logits_np = np.ndarray = raw_output[0][0]
            logits_tensor = torch.from_numpy(logits_np)
            probabilites = torch.nn.functional.softmax(logits_tensor, dim=0)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilites,0)

            confidence_value = confidence_tensor.item()
            predicted_idx_value = predicted_idx_tensor.item()
            predicted_class = self.class_names[predicted_idx_value]

            result = {
                    "class": predicted_class,
                    "confidence": round(confidence_value,4)
            }
            return result

        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": "Failed to process", "details": str(e)}
