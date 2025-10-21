# PneumoDetect: Chest X-Ray Analysis

![PneumoDetect Logo](./assets/Readme-Logo.png)

PneumoDetect is a full-stack web application designed to classify chest X-ray images for the detection of pneumonia. It leverages a powerful deep learning model served via a high-performance FastAPI backend, with a clean, responsive front end for user interaction. The entire application is containerized with Docker for easy setup and consistent deployment.

## Features

-   **FastAPI Backend:** A robust API built with FastAPI to handle image processing and prediction requests.
-   **Efficient ML Inference:** Utilizes a pre-trained `EfficientNetV2-S` model converted to the ONNX format for fast, CPU-based inference.
-   **Clean & Responsive Frontend:** A simple user interface built with HTML, CSS, and vanilla JavaScript that works on both desktop and mobile.
-   **Fully Containerized:** The entire application stack (Frontend Web Server & Backend API) is managed by Docker and Docker Compose for one-command setup.
-   **Scalable Architecture:** The separate frontend (Nginx) and backend (Uvicorn) containers represent a production-ready pattern that can be scaled independently.

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language. |
| | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) | High-performance web framework for the API. |
| | **ONNX Runtime** | For efficient, cross-platform model inference. |
| **Frontend**| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) | Structure of the web application. |
| | ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) | Styling the user interface. |
| | ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) | Handling user interaction and API calls. |
| **Deployment**| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) | Containerization of the application. |
| | **Docker Compose** | Orchestrating the multi-container setup. |
| | ![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white) | Serving the frontend and acting as a reverse proxy. |

## Project Structure

The project is organized into a clean, scalable structure:

```
.
├── api/                  # Contains all backend FastAPI code
│   ├── Dockerfile        # Blueprint for the backend container
│   └── main.py           # Main FastAPI application logic
│   └── pipeline.py       # Handles the ONNX model inference
├── frontend/             # Contains all frontend code and assets
│   ├── assets/           # Logos and other static images
│   ├── css/
│   ├── js/
│   ├── Dockerfile        # Blueprint for the frontend Nginx container
│   ├── index.html        # Main HTML file
│   └── nginx.conf        # Nginx configuration for serving files and proxying API
├── saved_models/         # Stores the trained .onnx model file
├── .dockerignore         # Specifies files to ignore during Docker builds
├── docker-compose.yml    # Defines how to run the frontend and backend together
└── requirements.txt      # Python dependencies for the backend
```

---

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

You must have the following software installed:

-   **Docker:** [Get Docker](https://docs.docker.com/get-docker/)
-   **Docker Compose:** (Usually included with Docker Desktop)
-   **Git:** [Get Git](https://git-scm.com/downloads)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Check `requirements.txt`:**
    Ensure this file exists and contains the necessary Python libraries (e.g., `fastapi`, `uvicorn`, `onnxruntime`, `torchvision`). If not, you may need to generate it from your Python environment.

---

## Running the Application

With Docker and Docker Compose, running the entire application is a single command.

1.  **Build and run the containers:**
    From the root directory of the project, run the following command:
    ```bash
    docker-compose up --build
    ```
    This command will build and start the `frontend` and `backend` containers.

2.  **Access the application:**
    Once the containers are running, open your web browser and navigate to:
    **[http://localhost:8080](http://localhost:8080)**

3.  **Stopping the application:**
    Press `Ctrl + C` in the terminal, then run `docker-compose down` to stop and remove the containers.

## How to Use

1.  Click the **"Choose an Image"** button.
2.  Select a chest X-ray image (`.jpeg` or `.png`) from your computer.
3.  A preview of the selected image will be displayed.
4.  Click the **"Predict"** button.
5.  The application will show a loading spinner while the backend processes the image.
6.  The result, including the predicted class (**NORMAL** or **PNEUMONIA**) and the confidence score, will be displayed.

---

## API Endpoint Documentation

The API is not directly exposed to the host machine. All requests must go through the Nginx proxy running on port `8080`.

### `/predict`

-   **Method:** `POST`
-   **Description:** Uploads an image file for pneumonia classification.
-   **URL:** `http://localhost:8080/predict`
-   **Body:** `multipart/form-data` with a key named `file`.

**Example using `curl`:**
```bash
curl -X POST -F "file=@/path/to/your/xray.jpeg" http://localhost:8080/predict
```
*(Note: This command sends the request to the frontend Nginx container, which then securely proxies it to the backend API container.)*

#### For Backend Developers
If you need to test the API directly (bypassing the Nginx proxy), you can temporarily uncomment the `ports` section under the `backend` service in your `docker-compose.yml` file and rebuild:
```yaml
# In docker-compose.yml
services:
  backend:
    # ...
    ports:          # Add a port section in the Docker-Compose yaml file
      - "8000:8000" # create a suitable port to expose the API
```
After running `docker-compose up --build`, you can then use `http://localhost:8000/predict`.

#### Success Response (200 OK)
```json
{
  "class": "PNEUMONIA",
  "confidence": 0.9876
}
```

#### Error Response (400 Bad Request)
If the file is not an image:
```json
{
  "detail": "File is not an image. Uploaded file type is: application/pdf"
}
