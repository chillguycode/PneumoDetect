from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import InfPipeline
from pathlib import Path
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on application startup
    print("--- Running app startup logic ---")
    model_path = Path(__file__).parent.parent / "saved_models" / "efficientnet_v2_s_best.onnx"
    ml_models["pipeline"] = InfPipeline(model_path=str(model_path))
    print("--- Startup complete ---")
    yield
    # Code to run on application shutdown
    print("--- Running app shutdown logic ---")
    ml_models.clear()
    print("--- Shutdown complete ---")

app = FastAPI(title='PneumoDetect API', lifespan=lifespan)

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


def get_pipeline():
    return ml_models["pipeline"]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pipeline: InfPipeline = Depends(get_pipeline)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File is not an image. Uploaded file type is: {file.content_type}"
        )
    try:
        image_bytes = await file.read()
        result = pipeline.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Step 4: If there was no error, return the successful result.
    return result
