from fastapi.testclient import TestClient


# Test 1: Successful prediction
def test_predict_endpoint_success(test_client: TestClient, valid_image_bytes: bytes):
    files = {'file': ('test_image.png', valid_image_bytes, 'image/png')}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 200
    response_data = response.json()
    assert "class" in response_data
    assert "confidence" in response_data


# Test 2: Uploading a non-image file
def test_predict_endpoint_not_an_image(test_client: TestClient, non_image_bytes: bytes):
    files = {'file': ('test.txt', non_image_bytes, 'text/plain')}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 400
    assert "File is not an image" in response.json()["detail"]


# Test 3: The pipeline encounters an internal error
def test_predict_endpoint_pipeline_error(test_client: TestClient, pipeline_instance, valid_image_bytes, mocker):
    mocked_error = {"error": "Model loading failed", "details": "Specific error"}
    mocker.patch.object(pipeline_instance, 'predict', return_value=mocked_error)

    files = {'file': ('test_image.png', valid_image_bytes, 'image/png')}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Model loading failed"


# Test 4: Submitting the form with no file
def test_predict_endpoint_no_file(test_client: TestClient):
    response = test_client.post("/predict")
    assert response.status_code == 422
