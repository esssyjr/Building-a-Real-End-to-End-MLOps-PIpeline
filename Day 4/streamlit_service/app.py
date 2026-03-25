import streamlit as st
import requests
from PIL import Image

# =============================
# App Config
# =============================
st.set_page_config(
    page_title="Naira Note Classifier",
    page_icon="💵",
    layout="centered"
)

API_URL = "http://fastapi:8020/predict"


# =============================
# UI Header
# =============================
st.title("💵 Naira Note Classification")
st.markdown(
    "Upload an image of a **Nigerian Naira note** to classify its denomination."
)

st.divider()

# =============================
# File Upload
# =============================
uploaded_file = st.file_uploader(
    "Upload a Naira note image",
    type=["jpg", "jpeg", "png"]
)

# =============================
# Prediction
# =============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify"):
        with st.spinner("Running model inference..."):
            response = requests.post(
                API_URL,
                files={"file": uploaded_file.getvalue()},
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()

            st.success(f"### ₦{result['predicted_class']}")
            st.metric(
                label="Confidence",
                value=f"{result['confidence']:.2%}"
            )

            with st.expander("Model Information"):
                st.write(f"**Model Name:** {result['model_name']}")
                st.write(f"**Alias:** {result['model_alias']}")
                st.write(f"**Version:** {result['model_version']}")

        else:
            st.error("❌ Prediction failed. Please try again.")

# =============================
# Footer
# =============================
st.divider()
st.caption("Powered by EJAZTECH.AI • MLflow • FastAPI • Streamlit")
