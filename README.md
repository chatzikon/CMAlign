# CMAlign
# Multimodal Image API (FastAPI + Docker)

## 📌 Overview

This project implements a multimodal application that supports:

* **Image-to-Text**: generating a caption from an input image
* **Text-to-Image**: generating an image from a textual prompt

The application is built using **FastAPI** and exposes REST endpoints that can be accessed programmatically. It is containerized using Docker for portability and ease of deployment.

---

## 🧱 Project Structure

```
project/
├── api/
│   ├── __init__.py
│   └── server.py
├── models/
│   ├── __init__.py
│   ├── components.py
│   ├── discriminator.py
│   ├── image_to_text.py
│   ├── load_model.py
│   ├── txt_to_image.py
│   └── vae.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ⚙️ Features

* REST API built with FastAPI
* Bearer token authentication
* Image upload and processing
* Model inference for both modalities
* Dockerized deployment
* Configurable port exposure

---

## 🔐 Authentication

All endpoints are protected using **Bearer Token Authentication**.

Each request must include:

```
Authorization: Bearer <your-token>
```

The token is configured via environment variable:

```
API_BEARER_TOKEN=my-secret-token
```

---

## 🚀 Running with Docker

### 1️⃣ Build and run the container

From the project root:

```bash
docker compose up --build
```

---

### 2️⃣ Access the API

By default, the application is exposed on:

```
http://localhost:4523
```

Interactive API documentation:

```
http://localhost:4523/docs
```

---

## 📡 API Endpoints

### 🔹 GET `/`

Health check (requires authentication)

---

### 🔹 POST `/image-to-text`

**Description:** Generates a caption from an image

**Request:**

* Content-Type: `multipart/form-data`
* Field: `file` (image)

**Example (Python):**

```python
import requests

headers = {
    "Authorization": "Bearer my-secret-token"
}

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:4523/image-to-text",
        files={"file": f},
        headers=headers
    )

print(response.json())
```

---

### 🔹 POST `/text-to-image`

**Description:** Generates an image from text

**Request:**

```json
{
  "prompt": "a cat sitting on a chair"
}
```

**Example (Python):**

```python
import requests

headers = {
    "Authorization": "Bearer my-secret-token"
}

response = requests.post(
    "http://localhost:4523/text-to-image",
    json={"prompt": "a cat"},
    headers=headers
)

with open("output.png", "wb") as f:
    f.write(response.content)
```

---

## ⚡ Performance Optimization

The model is loaded **once at application startup** and reused across requests.
This significantly improves performance compared to loading the model per request.

---

## 🐳 Docker Details

* Base image: `python:3.12-slim`
* Internal port: `4523`
* External port: `4523`
* Dependencies installed via `requirements.txt`

---

## 🔧 Environment Variables

| Variable           | Description          | Default           |
| ------------------ | -------------------- | ----------------- |
| `API_BEARER_TOKEN` | Authentication token | `my-secret-token` |

---

## ⚠️ Notes

* The current setup uses a simple shared token for authentication.
* HTTPS and advanced authentication (e.g., OpenID Connect) can be added in future improvements.
* Model loading may take time during container startup.

---

## 📄 License

This project is intended for academic and experimental use.
