from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

def test_predict_endpoint_success(test_client: TestClient, valid_image_bytes: bytes, mocker: MockerFixture):
    mocked_result = {"class": "PNEUMONIA", "confidence": 0.998}
    mocker.patch("api.main.pipeline.predict", return_value = mocked_result)

    files = {'file': ('test_image.png', valid_image_bytes, 'image/png')}
    response = test_client.post("/predict", files=files)




def test_predict_endpoint_pipeline_error(test_client: TestClient, valid_image_bytes: bytes, mocker: MockerFixture):
    mocked_error = {"error": "Model loading failed", "details":"Specific error"}
    mocker.patch("api.main.pipeline.predict", return_value = mocked_error)

    files = {'file': ('test_image.png', valid_image_bytes, 'image/png')}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Model Loading Failed"



def test_predict_endpoint_no_file(test_client: TestClient):
    response = test_client.post("/predict")
    assert response.status_code == 422
