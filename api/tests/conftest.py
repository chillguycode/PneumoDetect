from api.tests.create_dummy_model import generate_dummy_model
import pytest
from pathlib import Path
from PIL import Image
import io


from fastapi.testclient import TestClient
from ..main import app,InfPipeline


@pytest.fixture(scope="session")
def dummy_onnx_path() -> Path:
    model_dir = Path(__file__).parent.parent.parent/"saved_models_test"
    model_path = model_dir/"dummy_model.onnx"
    generate_dummy_model(model_path)
    return model_path


@pytest.fixture
def dummy_onnx_path() -> Path:
    model_dir = Path(__file__).parent.parent.parent/ "saved_model_test"
    model_path = model_dir/"dummy_model.onnx"
    generate_dummy_model(model_path)
    return model_path


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_image_bytes() -> bytes:
    image = Image.new('RGB', (100,100), color = 'red')
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    return byte_arr.getvalue()


@pytest.fixture
def non_image_bytes() -> bytes:
    return b"this is definitely not an image"
