import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import mlflow
from mlflow.tracking import MlflowClient
from threading import Lock

# =============================
# MLflow setup
# =============================
TRACKING_URL = "http://mlflow_service:5000"
MODEL_NAME = "NAIRA"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URL)
client = MlflowClient(tracking_uri=TRACKING_URL)

# =============================
# Storage paths (SHARED VOLUME)
# =============================
BASE_DATA_DIR = "/shared_data"

READY_DIR = os.path.join(BASE_DATA_DIR, "ready", "images")

os.makedirs(READY_DIR, exist_ok=True)

# =============================
# Global cached model state
# =============================
model = None
model_version = None
model_lock = Lock()

# =============================
# Class names
# =============================
class_names = ["10", "100", "1000", "20", "200", "5", "50", "500"]

# =============================
# Image preprocessing
# =============================
def prepare_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# =============================
# Alias-aware model loader
# =============================
def get_model():
    global model, model_version

    with model_lock:
        version_info = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS
        )

        if model is None or model_version != version_info.version:
            print(
                f"🔄 Reloading model: {MODEL_NAME} "
                f"(alias={MODEL_ALIAS}, version={version_info.version})"
            )

            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            model = mlflow.pyfunc.load_model(model_uri)
            model_version = version_info.version

    return model, model_version

# =============================
# Prediction function
# =============================
def predict(image_array):
    model, version = get_model()

    preds = model.predict(image_array)
    class_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    return class_names[class_index], confidence, version

# =============================
# Save inference sample (READY → images only)
# =============================
def save_ready_sample(image: Image.Image):
    sample_id = str(uuid.uuid4())
    image_path = os.path.join(READY_DIR, f"{sample_id}.jpg")
    image.save(image_path, format="JPEG")
    return sample_id

# =============================
# FastAPI app
# =============================
app = FastAPI(title="Naira Note Classification API")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_array = prepare_image(image)

    predicted_class, confidence, version = predict(image_array)

    sample_id = save_ready_sample(image=image)

    return {
        "id": sample_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": version,
        "annotation_status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
