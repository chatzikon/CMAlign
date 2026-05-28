from fastapi import FastAPI, UploadFile,  Depends, HTTPException, status
#from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import io
import os


from models.image_to_text import  img_to_txt

import datetime
from contextlib import asynccontextmanager

from models.showo_service import ShowoService
from models.Showo.training.utils import get_config

# ---- Globals for model ----
model = None
tokenizer = None

#config = get_config("configs/showo_demo_w_clip_vit_512x512.yaml")
config = get_config()


showo = ShowoService(config)

app = FastAPI(title="Multimodal Image API")

# # Use an environment variable in practice
# API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "my-secret-token")
#
# security = HTTPBearer()
#
# def verify_bearer_token(
#     credentials: HTTPAuthorizationCredentials = Depends(security),
# ) -> str:
#     token = credentials.credentials
#
#     if token != API_BEARER_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing bearer token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#
#     return token

# ---- Routes ----

# @app.get("/")
# async def root(token: str = Depends(verify_bearer_token)):
#     return {"message": "API is running and authenticated"}


@app.get("/")
async def root():
    return {"message": "API is running"}


# ---- Request model ----
class TextToImageRequest(BaseModel):
    prompt: str


@app.post("/image-to-text")
async def image_to_text(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    prompt_file = "/app/api/prompt.txt"

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    caption = showo.image_to_text(
        image,
        #question="Please describe this image in detail."
        question=prompt
    )

    date_iso8601=datetime.datetime.now().isoformat()


    return {
        "tool_name": "CMAlign",
        "text": {"caption": caption},
        "in_id": 'afr55',
        "in_filename": file.filename,
        "image": True,
        "store_misp": False,
        "description": {"frame_start": 1, "frame_end": 1, "text": caption},
        "date_iso8601": date_iso8601
    }
