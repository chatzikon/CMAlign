from fastapi import FastAPI, UploadFile,  Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import io
import os


from models.image_to_text import  img_to_txt

import datetime
from contextlib import asynccontextmanager

from models.showo2_service import Showo2Service
from models.Showo.show_o2.utils import get_config

# ---- Globals for model ----
model = None
tokenizer = None

#config = get_config("configs/showo_demo_w_clip_vit_512x512.yaml")
config = get_config()


showo = Showo2Service(config)

app = FastAPI(title="Multimodal Image API")

# Use an environment variable in practice
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "my-secret-token")

security = HTTPBearer()

def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials

    if token != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token

# ---- Routes ----

@app.get("/")
async def root(token: str = Depends(verify_bearer_token)):
    return {"message": "API is running and authenticated"}


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
        question=prompt,
        #question="Please describe this image in detail."
        # question="Analyze this image for investigation-relevant information. "
        #
        #          "Do NOT just describe objects."
        #          "Infer what the scene could mean from an investigative/security perspective. "
        #
        #          "For each finding, output: "
        #
        #          "observation, "
        #          "possible significance,"
        #          "risk level (low/medium/high), "
        #          "confidence (low/medium/high), "
        #
        #          "Rules: "
        #          "You may infer plausible threats, criminal activity, concealment, fraud, violence risk, cybercrime relevance,"
        #          " trafficking/resale indicators, etc.,"
        #          "Do not state speculation as fact but as inferred hypothesis.,"
        #          "Keep outputs concise.,"
        #          "Focus on what would matter to an investigator, analyst, or threat assessor.,"
        #
        #          "Output format:"
        #
        #          "Observation: ..."
        #          "Significance: ..."
        #          "Risk: ..."
        #          "Confidence: ...,"
        #
        #          "Example:"
        #
        #          "Observation: Multiple boxed phones with visible serial labels and cash"
        #          "Significance: May indicate resale activity or potentially stolen-property handling"
        #          "Risk: Medium"
        #          "Confidence: Medium"
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
