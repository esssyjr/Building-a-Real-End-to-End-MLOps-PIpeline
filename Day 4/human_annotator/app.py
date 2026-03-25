import streamlit as st
from PIL import Image
import os
import shutil
from datetime import datetime

# =============================
# Paths (shared volume)
# =============================
BASE_DIR = "/shared_data"

INCOMING_IMAGES = os.path.join(BASE_DIR, "ready/images")  
VALIDATED_IMAGES = os.path.join(BASE_DIR, "validated")

os.makedirs(VALIDATED_IMAGES, exist_ok=True)

# =============================
# App Config
# =============================
st.set_page_config(
    page_title="Human Annotation Interface",
    page_icon="🧑‍🏫",
    layout="centered"
)

st.title("🧑‍🏫 Human-in-the-Loop Annotation")
st.caption("Confirm or correct model predictions")

# =============================
# Load pending samples
# =============================
pending_images = sorted([
    f for f in os.listdir(INCOMING_IMAGES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not pending_images:
    st.success("✅ No pending images to annotate.")
    st.stop()

# =============================
# Select image
# =============================
selected_image = st.selectbox(
    "Select image to annotate",
    pending_images
)

image_path = os.path.join(INCOMING_IMAGES, selected_image)
image = Image.open(image_path).convert("RGB")
st.image(image, caption=selected_image, use_column_width=True)

# =============================
# Annotation form
# =============================
st.divider()
st.subheader("Annotation")

class_names = ["5", "10", "20", "50", "100", "200", "500", "1000"]

correct_class = st.selectbox(
    "Select correct denomination",
    class_names
)

confirm = st.checkbox("I confirm this annotation is correct")

# =============================
# Submit annotation
# =============================
if st.button("✅ Submit Annotation", disabled=not confirm):
    # Move image to validated folder and organize by class
    class_folder = os.path.join(VALIDATED_IMAGES, correct_class)
    os.makedirs(class_folder, exist_ok=True)

    shutil.move(
        image_path,
        os.path.join(class_folder, selected_image)
    )

    st.success("🎉 Annotation saved successfully!")
    st.experimental_rerun()
