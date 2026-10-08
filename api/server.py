from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

import datetime
import os
import secrets
import shutil
import tempfile
import zipfile

from pathlib import Path

from models.showo2_qwen3_service import (
    Showo2Qwen3Service,
)


# ==========================================================
# Configuration
# ==========================================================

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


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# This is NOT the maximum number of input images.
#
# Example:
#   1000 images
#
# are processed internally as:
#
#   128
#   128
#   ...
#   final remaining images
#
PROCESSING_CHUNK_SIZE = 128


# ==========================================================
# Model
# ==========================================================

showo = Showo2Qwen3Service(
    wan_vae_path=WAN_VAE_PATH,
    stage2_checkpoint=SHOWO_STAGE2_CHECKPOINT,
)


# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="Multimodal Image API"
)


# ==========================================================
# Authentication
# ==========================================================

CMALIGN_API_KEY = os.getenv(
    "CMALIGN_API_KEY"
)

if not CMALIGN_API_KEY:
    raise RuntimeError(
        "CMALIGN_API_KEY environment variable is not set"
    )


bearer_security = HTTPBearer(
    auto_error=False
)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_security
    ),
):

    if credentials is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    provided_key = credentials.credentials

    if not secrets.compare_digest(
        provided_key,
        CMALIGN_API_KEY,
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )


# ==========================================================
# Upload helpers
# ==========================================================

async def save_upload(
    upload: UploadFile,
    destination: Path,
):
    """
    Save an UploadFile to disk without loading
    the whole file into RAM.
    """

    await upload.seek(0)

    with destination.open("wb") as output_file:

        while True:

            chunk = await upload.read(
                1024 * 1024
            )

            if not chunk:
                break

            output_file.write(chunk)


async def prepare_request_images(
    files: list[UploadFile],
    tmp_dir: Path,
):
    """
    Convert all request inputs into image paths.

    Supports:

      - one image
      - multiple images
      - one ZIP
      - multiple ZIPs
      - images + ZIPs

    All files are kept inside tmp_dir and are
    deleted after the request completes.
    """

    upload_dir = (
        tmp_dir / "uploads"
    )

    extracted_dir = (
        tmp_dir / "extracted"
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_images = []

    # More than one uploaded object automatically
    # means batch response format.
    batch_mode = len(files) > 1


    for upload_index, upload in enumerate(
        files
    ):

        original_name = Path(
            upload.filename
            or f"upload_{upload_index}"
        ).name

        suffix = Path(
            original_name
        ).suffix.lower()


        # ==================================================
        # Normal image
        # ==================================================

        if suffix in IMAGE_EXTENSIONS:

            target = (
                upload_dir
                / (
                    f"{upload_index:06d}"
                    f"{suffix}"
                )
            )

            await save_upload(
                upload=upload,
                destination=target,
            )

            prepared_images.append(
                {
                    "path": target,
                    "name": original_name,
                }
            )

            continue


        # ==================================================
        # ZIP
        # ==================================================

        if suffix == ".zip":

            # A ZIP always uses the batch-style
            # response, even if it contains only
            # one image.
            batch_mode = True

            zip_path = (
                upload_dir
                / f"{upload_index:06d}.zip"
            )

            await save_upload(
                upload=upload,
                destination=zip_path,
            )

            try:

                with zipfile.ZipFile(
                    zip_path,
                    "r",
                ) as archive:

                    for (
                        member_index,
                        info,
                    ) in enumerate(
                        archive.infolist()
                    ):

                        # Ignore directories.
                        if info.is_dir():
                            continue

                        member_suffix = Path(
                            info.filename
                        ).suffix.lower()

                        # Ignore non-image files
                        # inside the ZIP.
                        if (
                            member_suffix
                            not in IMAGE_EXTENSIONS
                        ):
                            continue


                        # ----------------------------------
                        # Safe extraction
                        # ----------------------------------
                        #
                        # Do NOT use extract() or
                        # extractall().
                        #
                        # We create the destination name
                        # ourselves. Therefore something
                        # inside a malicious ZIP such as:
                        #
                        # ../../etc/something
                        #
                        # cannot escape tmp_dir.
                        # ----------------------------------

                        target = (
                            extracted_dir
                            / (
                                f"{upload_index:06d}_"
                                f"{member_index:06d}"
                                f"{member_suffix}"
                            )
                        )

                        with (
                            archive.open(
                                info,
                                "r",
                            ) as source,
                            target.open(
                                "wb"
                            ) as destination,
                        ):

                            shutil.copyfileobj(
                                source,
                                destination,
                            )


                        # Keep the original path/name
                        # from inside the ZIP so that
                        # the result remains identifiable.
                        prepared_images.append(
                            {
                                "path": target,
                                "name": info.filename,
                            }
                        )


            except zipfile.BadZipFile as exc:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid ZIP file: "
                        f"{original_name}"
                    ),
                ) from exc

            continue


        # ==================================================
        # Unsupported uploaded type
        # ==================================================

        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type: "
                f"{original_name}"
            ),
        )


    # ======================================================
    # Make sure something useful was actually supplied
    # ======================================================

    if not prepared_images:

        raise HTTPException(
            status_code=400,
            detail=(
                "No supported images were found "
                "in the request."
            ),
        )


    return (
        prepared_images,
        batch_mode,
    )


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
async def root():

    return {
        "message": "API is running"
    }


@app.get("/health")
async def health():

    return {
        "status": "ok"
    }


# ==========================================================
# Image -> Text
# ==========================================================

@app.post("/image-to-text")
async def image_to_text(
    files: list[UploadFile] = File(
        ...,
        alias="file",
    ),
    _api_key: str = Depends(
        verify_api_key
    ),
):

    # ======================================================
    # Every request gets its own temporary directory.
    #
    # Example:
    #
    # /tmp/cmalign_request_xxxxx/
    #
    # EVERYTHING associated with the request stays here.
    # ======================================================

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="cmalign_request_"
        )
    )


    try:

        # ==================================================
        # 1. Save uploads and extract ZIP files
        # ==================================================

        (
            prepared_images,
            batch_mode,
        ) = await prepare_request_images(
            files=files,
            tmp_dir=tmp_dir,
        )


        # ==================================================
        # 2. Read prompt
        # ==================================================

        with open(
            PROMPT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            prompt = f.read()


        # ==================================================
        # 3. Process images
        #
        # No maximum image count.
        #
        # images_to_text() internally processes
        # them in chunks.
        # ==================================================

        model_results = (
            showo.images_to_text(
                image_paths=[
                    image_info["path"]
                    for image_info
                    in prepared_images
                ],
                question=prompt,
                alpha=FUSION_ALPHA,
                max_new_tokens=384,
                chunk_size=
                    PROCESSING_CHUNK_SIZE,
            )
        )


        # ==================================================
        # 4. Construct output
        # ==================================================

        date_iso8601 = (
            datetime.datetime.now()
            .isoformat()
        )

        results = []


        for (
            image_info,
            model_result,
        ) in zip(
            prepared_images,
            model_results,
        ):

            # ----------------------------------------------
            # Successful image
            # ----------------------------------------------

            if (
                model_result["status"]
                == "success"
            ):

                caption = (
                    model_result["caption"]
                )

                results.append(
                    {
                        "in_filename":
                            image_info["name"],

                        "status":
                            "success",

                        "text": {
                            "caption":
                                caption,
                        },

                        "image":
                            True,

                        "store_misp":
                            False,

                        "description": {
                            "frame_start": 1,
                            "frame_end": 1,
                            "text": caption,
                        },
                    }
                )


            # ----------------------------------------------
            # Failed image
            # ----------------------------------------------

            else:

                results.append(
                    {
                        "in_filename":
                            image_info["name"],

                        "status":
                            "error",

                        "error":
                            model_result["error"],
                    }
                )


        # ==================================================
        # 5. ONE NORMAL IMAGE
        #
        # Preserve the exact old API structure.
        #
        # This is important for Jack's existing
        # integration.
        # ==================================================

        if (
            not batch_mode
            and len(results) == 1
        ):

            result = results[0]


            if (
                result["status"]
                == "error"
            ):

                raise HTTPException(
                    status_code=422,
                    detail=result["error"],
                )


            caption = (
                result["text"]["caption"]
            )


            return {
                "tool_name":
                    "CMAlign",

                "text": {
                    "caption":
                        caption,
                },

                "in_id":
                    "afr55",

                "in_filename":
                    result["in_filename"],

                "image":
                    True,

                "store_misp":
                    False,

                "description": {
                    "frame_start": 1,
                    "frame_end": 1,
                    "text": caption,
                },

                "date_iso8601":
                    date_iso8601,
            }


        # ==================================================
        # 6. MULTIPLE IMAGES / ZIP
        # ==================================================

        successful = sum(
            result["status"] == "success"
            for result in results
        )

        failed = (
            len(results)
            - successful
        )


        return {
            "tool_name":
                "CMAlign",

            "count":
                len(results),

            "successful":
                successful,

            "failed":
                failed,

            "results":
                results,

            "date_iso8601":
                date_iso8601,
        }


    finally:

        # ==================================================
        # GUARANTEED CLEANUP
        # ==================================================
        #
        # This happens after:
        #
        # ✓ successful inference
        # ✓ invalid ZIP
        # ✓ corrupt image
        # ✓ Show-o2 error
        # ✓ Qwen error
        # ✓ CUDA error
        # ✓ HTTPException
        # ✓ any unexpected exception
        #
        # Original uploads, ZIPs and extracted
        # images are all removed.
        # ==================================================

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )