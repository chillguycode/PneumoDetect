from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import InfPipeline # Assuming your inference file is named pipeline.py
from pathlib import Path
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Running app startup logic ---")
    
    saved_models_dir = Path(__file__).parent.parent / "saved_models"
    
    GUARD_MODEL_PATH = saved_models_dir / "cnn_guard_best.onnx"
    MAIN_MODEL_PATH = saved_models_dir / "efficientnet_v2_s_best.onnx" 
    
    ml_models["pipeline"] = InfPipeline(
        guard_model_path=str(GUARD_MODEL_PATH),
        main_model_path=str(MAIN_MODEL_PATH)
    )
    
    print("--- Startup complete ---")
    yield
    # Code to run on application shutdown
    print("--- Running app shutdown logic ---")
    ml_models.clear()
    print("--- Shutdown complete ---")

app = FastAPI(title='PneumoDetect (Guarded) API', lifespan=lifespan)

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
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> InfPipeline:
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
        raise HTTPException(status_code=500, detail=result["details"])

    if result.get("guard_status") == "FAILED":
        # Using 406 Not Acceptable status to signify input rejection
        raise HTTPException(
            status_code=406, 
            detail=result["message"]
        )

    return result
