import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8001"

app_icon = Image.open("UI/ensemble_logo.png")

#st.set_page_config(layout="wide")

st.set_page_config(
    page_title="CMAlign",
    page_icon=app_icon, # You can also add an emoji or a URL to a favicon here
    layout="wide"
)


# Header
# Header
header_col1, header_col2 = st.columns([4, 1]) # Adjust ratio as needed

with header_col1:
    st.markdown(
        """
        <h1 style='text-align: left;'>Cross-modal Align System</h1>
        <p style='text-align: left;'>Image ↔ Text Description Interface</p>
        """,
        unsafe_allow_html=True
    )

with header_col2:
    # Replace 'logo.png' with your local path or a URL
    st.image("UI/ensemble_logo.png", use_container_width=True)

st.divider()

# Sidebar
mode = st.sidebar.radio(
    "Select Task",
    #["Text → Image Generation", "Image → Text Description"]
["Image → Text Description"]
)

col1, col2 = st.columns(2)

# INPUT
with col1:

    st.subheader("Input")

    if mode == "Text → Image Generation":

        prompt = st.text_area(
            "Text Description",
            placeholder="Example: A futuristic city at sunset"
        )

    else:

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_image:
            st.image(uploaded_image)

    run = st.button("Run Model")

# OUTPUT
with col2:

    st.subheader("Output")

    if run:

        with st.spinner("Processing..."):

            if mode == "Text → Image Generation":

                response = requests.post(
                    f"{API_URL}/text-to-image",
                    json={"prompt": prompt}
                )



                image = Image.open(io.BytesIO(response.content))

                st.image(image)

            else:

                files = {"file": uploaded_image}

                response = requests.post(
                    f"{API_URL}/image-to-text",
                    files=files
                )

                caption = response.json()['text']['caption']
                print(caption)

                st.success(caption)