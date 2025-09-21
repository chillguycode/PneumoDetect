from fastapi import FastAPI, HTTPException, UploadFile, File
from .pipeline import InfPipeline
from pathlib import Path
from PIL import Image


app = FastAPI(title='PneumoDetect API')


MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "efficientnet_v2_s_best.onnx"


pipeline = InfPipeline(model_path=str(MODEL_PATH))

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
                status_code=400,
                detail=f"File is not an image. Uploaded file type is: {file.content_type}"
                )

    try:
        image_bytes = await file.read()
        result = pipeline.predict(image_bytes)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An unexpected error occurred: {str(e)}")
