import streamlit as st
import requests
from PIL import Image

# =========================
# Config
# =========================
API_URL = "http://127.0.0.1:8000/predict"


# =========================
# UI Title
# =========================
st.title("🐱 Cat vs Not Cat Classifier")
st.write("Upload an image and get prediction from the model")

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                # Send request to FastAPI
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    label = result["label"]
                    confidence = result["confidence"]

                    st.success(f"Prediction: {label}")
                    st.info(f"Confidence: {confidence:.4f}")

                else:
                    st.error("Error from API")
                    st.text(response.text)

            except Exception as e:
                st.error(f"Failed to connect to API: {e}")