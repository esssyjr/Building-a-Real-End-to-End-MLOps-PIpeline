# =============================
# Libraries Import
# =============================
import os
import shutil
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.models.signature import infer_signature

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime, timedelta

# =============================
# Constants
# =============================
SHARED_DATA = "/shared_data"
VALIDATED_DIR = os.path.join(SHARED_DATA, "validated")
TRAIN_DIR = "/opt/airflow/data/Train"
MIN_SAMPLES = 10

# Valid Naira classes
VALID_CLASSES = ["5", "10", "20", "50", "100", "200", "500", "1000"]

# =============================
# Default DAG arguments
# =============================
default_args = {
    "owner": "Ismail",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 11),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email": ["ismailismailtj@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
}

# =============================
# DAG Definition (EVENT-DRIVEN)
# =============================
dag = DAG(
    dag_id="naira_classification_human_in_loop_training",
    default_args=default_args,
    schedule=None,   # event-driven
    catchup=False,
    tags=["mlops", "human-in-loop"],
)

# =============================
# Check if enough validated data exists
# =============================
def validated_data_ready():
    if not os.path.exists(VALIDATED_DIR):
        return False

    total = 0
    for cls in os.listdir(VALIDATED_DIR):
        if cls not in VALID_CLASSES:
            continue  # ignore invalid folders
        cls_dir = os.path.join(VALIDATED_DIR, cls)
        if os.path.isdir(cls_dir):
            total += len([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])

    return total >= MIN_SAMPLES

# =============================
# Merge validated data into training dataset (and CLEAN UP)
# =============================
def merge_validated_into_train():
    if not os.path.exists(VALIDATED_DIR):
        return

    for cls in os.listdir(VALIDATED_DIR):
        if cls not in VALID_CLASSES:
            continue  # skip invalid folders

        src_cls_dir = os.path.join(VALIDATED_DIR, cls)
        dst_cls_dir = os.path.join(TRAIN_DIR, cls)

        if not os.path.isdir(src_cls_dir):
            continue

        os.makedirs(dst_cls_dir, exist_ok=True)

        for fname in os.listdir(src_cls_dir):
            if fname.lower().endswith((".jpg", ".png")):
                src = os.path.join(src_cls_dir, fname)
                dst = os.path.join(dst_cls_dir, fname)
                try:
                    # ✅ Updated: We use move instead of copy2
                    # This removes the file from VALIDATED_DIR and puts it in TRAIN_DIR
                    shutil.move(src, dst)
                    print(f"✅ Moved: {src} → {dst}")
                except Exception as e:
                    print(f"❌ Failed to move {src} → {dst}: {e}")

        # ✅ Updated: Clean up class folders once all files are moved
        try:
            if not os.listdir(src_cls_dir):
                os.rmdir(src_cls_dir)
                print(f"🗑️ Removed empty class folder: {src_cls_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning folder {src_cls_dir}: {e}")

# =============================
# Training function
# =============================
def train_naira_model(**kwargs):
    TRACKING_URL = "http://mlflow_service:5000"
    EXPERIMENT_NAME = "NAIRA"

    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.tensorflow.autolog(log_models=False)

    train_dir = TRAIN_DIR
    val_dir = "/opt/airflow/data/val"

    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )

    data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_dataset = data_aug.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32
    )

    val_dataset = data.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32
    )

    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    mobilenet.trainable = False

    model = tf.keras.Sequential([
        mobilenet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(VALID_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    with mlflow.start_run(run_name="mobilenetv2-human-in-loop"):
        model.fit(train_dataset, validation_data=val_dataset, epochs=1)

        x_example, _ = next(train_dataset)
        x_example = x_example.astype(np.float32)
        y_example = model.predict(x_example)

        signature = infer_signature(x_example, y_example)

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="NAIRA",
            signature=signature,
            input_example=x_example[:1]
        )

# =============================
# Tasks
# =============================
wait_for_validated_data = PythonSensor(
    task_id="wait_for_validated_data",
    python_callable=validated_data_ready,
    poke_interval=60,
    timeout=60 * 60 * 24,
    mode="poke",
    dag=dag,
)

merge_task = PythonOperator(
    task_id="merge_validated_data_into_train",
    python_callable=merge_validated_into_train,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_naira_model_task",
    python_callable=train_naira_model,
    dag=dag,
)

email_task = EmailOperator(
    task_id="send_training_success_email",
    to="ismailismailtj@gmail.com",
    subject="Naira HITL Training Completed ✅",
    html_content="""
    <h3>Human-in-the-Loop Training Completed</h3>
    <p>The Naira model has been retrained using human-validated data.</p>
    <p><b>Trigger:</b> ≥ 10 validated samples</p>
    <p><b>Tracking:</b> MLflow</p>
    <p>Date: {{ ds }}</p>
    """,
    dag=dag,
)

# =============================
# Dependency
# =============================
wait_for_validated_data >> merge_task >> train_task >> email_task
