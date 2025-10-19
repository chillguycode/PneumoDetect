
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import InfPipeline
from pathlib import Path


app = FastAPI(title='PneumoDetect API')

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


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


app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
