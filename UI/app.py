import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

def display_rag_results(text):
    if not text.get("rag_enabled", False):
        st.success(text.get("caption", ""))
        return

    st.subheader("Image Description")
    st.write(text.get("image_description", ""))

    st.subheader("Investigative Findings")

    findings = text.get("investigative_findings", [])

    if not findings:
        st.info("No applicable knowledge entries identified.")

    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
        "Not assessable": "gray",
    }

    for finding in findings:
        with st.container(border=True):
            st.markdown(
                f"**{finding['vr_id']} — {finding['vr_title']}**"
            )

            st.markdown("**Observation**")
            st.write(finding["observation"])

            st.markdown("**Significance**")
            st.write(finding["significance"])

            col1, col2 = st.columns(2)

            risk = finding["risk"]
            color = risk_colors.get(risk, "gray")

            col1.markdown(f"**Risk:** :{color}[{risk}]")
            col2.markdown(f"**Confidence:** {finding['confidence']}")

    with st.expander("Retrieved Knowledge"):
        st.json(text.get("retrieved_knowledge", []))

    with st.expander("RAG Provenance"):
        st.json(text.get("rag_provenance", {}))

    for warning in text.get("structured_validation_warnings", []):
        st.warning(warning)


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)

CMALIGN_API_KEY = os.getenv(
    "CMALIGN_API_KEY",
)


if not CMALIGN_API_KEY:
    raise RuntimeError(
        "CMALIGN_API_KEY environment variable is not set"
    )

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

UPLOAD_EXTENSIONS = [
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tif",
    "tiff",
    "zip",
]


app_icon = Image.open(
    "UI/ensemble_logo.png"
)

st.set_page_config(
    page_title="CMAlign",
    page_icon=app_icon,
    layout="wide",
)


# ==========================================================
# Header
# ==========================================================

header_col1, header_col2 = st.columns(
    [4, 1]
)

with header_col1:

    st.markdown(
        """
        <h1 style='text-align: left;'>
            Cross-modal Align System
        </h1>

        <p style='text-align: left;'>
            Image ↔ Text Description Interface
        </p>
        """,
        unsafe_allow_html=True,
    )


with header_col2:

    st.image(
        "UI/ensemble_logo.png",
        use_container_width=True,
    )


st.divider()


# ==========================================================
# Sidebar
# ==========================================================

mode = st.sidebar.radio(
    "Select Task",
    [
        "Image → Text Description",
    ],
)


col1, col2 = st.columns(2)


# ==========================================================
# INPUT
# ==========================================================

with col1:

    st.subheader("Input")

    uploaded_files = st.file_uploader(
        "Upload image(s) or ZIP archive(s)",
        type=UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} file(s) selected"
        )

        # ------------------------------------------
        # Preview only a small number of images.
        #
        # We deliberately do not try to preview
        # every image because a user may select
        # hundreds or thousands of files.
        # ------------------------------------------

        preview_limit = 8
        previewed = 0

        for uploaded_file in uploaded_files:

            suffix = Path(
                uploaded_file.name
            ).suffix.lower()

            if (
                suffix in IMAGE_EXTENSIONS
                and previewed < preview_limit
            ):

                try:

                    uploaded_file.seek(0)

                    preview_image = Image.open(
                        uploaded_file
                    )

                    st.image(
                        preview_image,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

                    previewed += 1

                except Exception:

                    st.warning(
                        f"Could not preview "
                        f"{uploaded_file.name}"
                    )

                finally:

                    uploaded_file.seek(0)

            elif suffix == ".zip":

                st.info(
                    f"ZIP archive selected: "
                    f"{uploaded_file.name}"
                )

        image_count = sum(
            Path(f.name).suffix.lower()
            in IMAGE_EXTENSIONS
            for f in uploaded_files
        )

        if image_count > preview_limit:

            st.caption(
                f"Showing the first "
                f"{preview_limit} image previews."
            )

    run = st.button(
        "Run Model",
        disabled=not uploaded_files,
    )


# ==========================================================
# OUTPUT
# ==========================================================

with col2:

    st.subheader("Output")

    if run:

        if not uploaded_files:

            st.warning(
                "Please upload at least one "
                "image or ZIP archive."
            )

        else:

            with st.spinner(
                "Processing..."
            ):

                try:

                    # ======================================
                    # Build multipart request.
                    #
                    # Important:
                    # Every uploaded object uses the SAME
                    # multipart field name: "file".
                    #
                    # This lets the API receive:
                    #
                    # file=image1.jpg
                    # file=image2.jpg
                    # file=images.zip
                    # ======================================

                    multipart_files = []

                    for uploaded_file in uploaded_files:

                        uploaded_file.seek(0)

                        mime_type = (
                            uploaded_file.type
                            or "application/octet-stream"
                        )

                        multipart_files.append(
                            (
                                "file",
                                (
                                    uploaded_file.name,
                                    uploaded_file,
                                    mime_type,
                                ),
                            )
                        )

                    # ======================================
                    # Authorization
                    # ======================================

                    headers = {}

                    headers = {
                        "Authorization": (
                            f"Bearer {CMALIGN_API_KEY}"
                        )
                    }

                    # ======================================
                    # Request
                    # ======================================

                    response = requests.post(
                        f"{API_URL}/image-to-text",
                        files=multipart_files,
                        headers=headers,
                    )

                    # Raise for 4xx / 5xx responses.
                    response.raise_for_status()

                    data = response.json()


                    # ======================================
                    # SINGLE IMAGE RESPONSE
                    #
                    # Keep compatibility with the original
                    # API response.
                    # ======================================

                    if "results" not in data:

                        caption = (
                            data["text"]["caption"]
                        )

                        filename = data.get(
                            "in_filename",
                            "Image",
                        )

                        st.markdown(
                            f"### {filename}"
                        )

                        display_rag_results(data["text"])

                    # ======================================
                    # MULTIPLE IMAGES / ZIP RESPONSE
                    # ======================================

                    else:

                        total = data.get(
                            "count",
                            len(data["results"]),
                        )

                        successful = data.get(
                            "successful",
                            0,
                        )

                        failed = data.get(
                            "failed",
                            0,
                        )

                        st.markdown(
                            "### Processing complete"
                        )

                        st.write(
                            f"Processed: **{total}**  "
                            f"| Successful: "
                            f"**{successful}**  "
                            f"| Failed: "
                            f"**{failed}**"
                        )

                        st.divider()

                        for index, result in enumerate(
                            data["results"],
                            start=1,
                        ):

                            filename = result.get(
                                "in_filename",
                                f"Image {index}",
                            )

                            status_value = result.get(
                                "status",
                                "success",
                            )

                            st.markdown(
                                f"#### {index}. "
                                f"{filename}"
                            )

                            if (
                                status_value
                                == "success"
                            ):

                                caption = (
                                    result
                                    .get(
                                        "text",
                                        {},
                                    )
                                    .get(
                                        "caption",
                                        "",
                                    )
                                )

                                display_rag_results(result["text"])

                            else:

                                error = result.get(
                                    "error",
                                    "Unknown processing error",
                                )

                                st.error(
                                    error
                                )

                except requests.HTTPError as exc:

                    try:

                        error_detail = (
                            response
                            .json()
                            .get(
                                "detail",
                                str(exc),
                            )
                        )

                    except Exception:

                        error_detail = str(exc)

                    st.error(
                        f"API error: "
                        f"{error_detail}"
                    )

                except requests.RequestException as exc:

                    st.error(
                        f"Could not connect "
                        f"to CMAlign API: {exc}"
                    )

                except Exception as exc:

                    st.error(
                        f"Unexpected error: {exc}"
                    )