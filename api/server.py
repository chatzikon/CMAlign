from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from PIL import Image
import io
import os
from pathlib import Path

import datetime
from contextlib import asynccontextmanager

from models.showo2_qwen3_service import Showo2Qwen3Service

PROMPT_FILE = (
    Path(__file__).resolve().parent
    / "prompt_init.txt"
)

WAN_VAE_PATH = os.getenv(
    "WAN_VAE_PATH",
    "checkpoints/Wan2.1_VAE.pth",
)

SHOWO_STAGE2_CHECKPOINT = os.getenv(
    "SHOWO_STAGE2_CHECKPOINT",
    "checkpoints/showo2_qwen3_stage2.pt",
)

FUSION_ALPHA = float(
    os.getenv(
        "FUSION_ALPHA",
        "0.5",
    )
)

showo = Showo2Qwen3Service(
    wan_vae_path=WAN_VAE_PATH,
    stage2_checkpoint=SHOWO_STAGE2_CHECKPOINT,
)

app = FastAPI(title="Multimodal Image API")

# ---- Routes ----

@app.get("/")
async def root():
    return {
        "message": "API is running"
    }


# ---- Request model ----
class TextToImageRequest(BaseModel):
    prompt: str


@app.post("/image-to-text")
async def image_to_text(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    with open(
            PROMPT_FILE,
            "r",
            encoding="utf-8",
    ) as f:
        prompt = f.read()

    final_analysis = showo.image_to_text(
        image=image,
        question=prompt,
        alpha=FUSION_ALPHA,
        max_new_tokens=384,
    )



    date_iso8601=datetime.datetime.now().isoformat()


    return {
        "tool_name": "CMAlign",
        "text": {
            #"visual_observations": caption1,
            "caption": final_analysis,
        },
        "in_id": 'afr55',
        "in_filename": file.filename,
        "image": True,
        "store_misp": False,
        "description": {"frame_start": 1, "frame_end": 1, "text": final_analysis},
        "date_iso8601": date_iso8601
    }
