# Runtime
FROM python:3.13-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api /app/api
COPY ./saved_models/cnn_guard_best.onnx /app/saved_models/
COPY ./saved_models/efficientnet_v2_s_best.onnx /app/saved_models/

EXPOSE 8000

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
