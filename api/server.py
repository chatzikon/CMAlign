from fastapi import FastAPI, UploadFile,  Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import io
import os


from models.load_model import load_model
from models.image_to_text import  img_to_txt
from models.text_to_image import  txt_to_img
from torchvision import transforms

from fastapi import Response
import datetime
from contextlib import asynccontextmanager

# ---- Globals for model ----
model = None
tokenizer = None



# ---- Lifespan (load model once) ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    print("Loading model...")
    model, tokenizer = load_model()

    print("MODEL TYPE:", type(model))
    print("TOKENIZER TYPE:", type(tokenizer))

    if model is None:
        raise RuntimeError("❌ Model is None → load_model() failed")

    yield

app = FastAPI(lifespan=lifespan, title="Multimodal Image API")

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
async def image_to_text(
    file: UploadFile,
    token: str = Depends(verify_bearer_token),
):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image_t=transform(image)
    # Call your model here
    #model,tokenizer=load_model()

    caption=img_to_txt(image_t,model,tokenizer)

    date_iso8601=datetime.datetime.now().isoformat()

    #caption = "A dog running through a grassy field."

    return {"tool_name":"CMAlign", "store_misp": False,
            "text": {"caption": caption}, "filename": file.filename, "date_iso8601": date_iso8601}


@app.post("/text-to-image")
async def text_to_image(
    data: dict,
    token: str = Depends(verify_bearer_token),
):

    prompt = data["prompt"]

    # Call your model here
    #model, tokenizer = load_model()

    image=txt_to_img(model, tokenizer, prompt)

    image_t = transforms.ToPILImage()(image.squeeze(0))

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        image_t.save(buf,  format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="test.png"'}

    return Response(im_bytes, headers=headers, media_type='image/png')