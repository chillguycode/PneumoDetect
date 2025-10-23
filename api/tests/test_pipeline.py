from torch.cuda import reset_accumulated_host_memory_stats
from api.tests.conftest import non_image_bytes
import pytest
from ..pipeline import InfPipeline
from PIL import Image
import io


def test_pipeline_intialization(pipeline_instance: InfPipeline):
    assert pipeline_instance is not None
    assert pipeline_instance.session is not None
    assert pipeline_instance.class_names == ["NORMAL", "PNEUMONIA"]


def test_pipeline_init_fails_with_bad_path():
    with pytest.raises(Exception):
        InfPipeline(model_path="non_existent/path/model.onnx")


def test_predict_success(pipeline_instance: InfPipeline, valid_image_bytes: bytes):
    result = pipeline_instance.predict(valid_image_bytes)
    assert "error" not in result
    assert "clas" in result
    assert "confidence" in result
    assert result["class"] in pipeline_instance.class_names
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_on_grayscale_image(pipeline_instance: InfPipeline):
    image = Image.new('L', (100,100), color = 'white')
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    image_bytes = byte_arr.getvalue()

    result = pipeline_instance.predict(image_bytes)
    assert "error" not in result
    assert "class" in result


def test_predict_invalid_bytes(pipeline_instance: InfPipeline):
    result = pipeline_instance.predict(non_image_bytes)
    assert "error" in result
    assert "details" in result
    assert result["error"] == "Failed to process"
