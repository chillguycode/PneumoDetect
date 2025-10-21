# Runtime
FROM python:3.13-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api /app/api
COPY ./saved_models /app/saved_models

EXPOSE 8000

CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
