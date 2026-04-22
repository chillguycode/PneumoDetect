import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import torch
import torch.nn.functional as F 
from api.eigencam import create_eigencam_session, EIGENCAM_TARGET_NODE, get_feature_maps, compute_eigencam, upsample_heatmap, render_heatmap
import base64

class InfPipeline:
    def __init__(self, guard_model_path: str, main_model_path: str):
        self.GUARD_THRESHOLD = 0.80
        self.GUARD_CLASS_NAMES = ["NOT_CHEST_XRAY", "CHEST_XRAY"] 
        self.MAIN_CLASS_NAMES = ["NORMAL", "PNEUMONIA"] 

        print(f"Loading guard model from: {guard_model_path}")
        self.guard_session = onnxruntime.InferenceSession(guard_model_path, providers=["CPUExecutionProvider"])
        self.guard_input_name = self.guard_session.get_inputs()[0].name
        print(f"Guard Model input name: {self.guard_input_name}")

        print(f"Loading Main Model from: {main_model_path}")
        self.main_session = onnxruntime.InferenceSession(main_model_path, providers=['CPUExecutionProvider'])
        self.main_input_name = self.main_session.get_inputs()[0].name
        print(f"Main Model input name: {self.main_input_name}")

        self.display_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ])
        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            ])

        print("Inference pipeline initialised successfully!")


        self.eigencam_session = create_eigencam_session(main_model_path,EIGENCAM_TARGET_NODE)

    def _process_image(self, image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
        """Loads and transforms image bytes into a batch tensor."""
        image: Image.Image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        display_image:Image.Image = self.display_transforms(image)
        input_tensor: torch.Tensor = self.inference_transforms(display_image).unsqueeze(0)
        # Add batch dimension (B, C, H, W)
        return input_tensor, display_image

    def _run_onnx_inference(self, session: onnxruntime.InferenceSession, input_name: str, input_batch: torch.Tensor) -> tuple[float, int]:
        """Runs inference and calculates the max confidence and predicted index."""
        
        onnx_input = {input_name: input_batch.numpy()}
        raw_output: list[np.ndarray] = session.run(None, onnx_input)
        
        logits_np = raw_output[0][0] 
        logits_tensor = torch.from_numpy(logits_np)
        
        probabilities = F.softmax(logits_tensor, dim=0) 
        
        confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 0)

        return confidence_tensor.item(), predicted_idx_tensor.item()

    def predict(self, image_bytes:bytes) -> dict:
        try:
            # Step 1: Process the image
            input_batch, display_image = self._process_image(image_bytes)

            # Step 2: GUARD MODEL CHECK
            guard_confidence, guard_idx = self._run_onnx_inference(
                self.guard_session, self.guard_input_name, input_batch
            )
            guard_class = self.GUARD_CLASS_NAMES[guard_idx]

            # Check if the Guard Model is confident it's a Chest X-ray
            if guard_class == "CHEST_XRAY" and guard_confidence >= self.GUARD_THRESHOLD:
                # Step 3: MAIN MODEL PREDICTION
                main_confidence, main_idx = self._run_onnx_inference(
                    self.main_session, self.main_input_name, input_batch
                )

                # Get Feature Maps
                feature_maps = get_feature_maps(self.eigencam_session, input_batch.numpy(), EIGENCAM_TARGET_NODE)
                activation_map = compute_eigencam(feature_maps[0])
                heatmap = upsample_heatmap(activation_map, (224,224))
                heatmap_image = render_heatmap(heatmap)
                
                main_class = self.MAIN_CLASS_NAMES[main_idx]

                buffer = io.BytesIO()
                display_image.save(buffer, format="PNG")
                buffer.seek(0)
                xray_b64= base64.b64encode(buffer.read()).decode("utf-8")

                return {
                    "guard_status": "PASSED",
                    "guard_confidence": round(guard_confidence, 4),
                    "prediction": main_class,
                    "confidence": round(main_confidence, 4),
                    "heatmap": heatmap_image,
                    "xray": xray_b64
                }
            else:
                # GUARD CHECK FAILED
                return {
                    "guard_status": "FAILED",
                    "guard_class": guard_class,
                    "guard_confidence": round(guard_confidence, 4),
                    "message": "Input image is not classified as a Chest X-ray with 95%+ confidence. Aborting main prediction."
                }

        except Exception as e:
            # Catch errors during image processing or inference
            print(f"Error during prediction: {e}")
            return {"error": "Failed to process image or run inference", "details": str(e)}
