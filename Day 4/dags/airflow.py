# =============================
# Libraries Import
# =============================
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta

# =============================
# Default DAG arguments
# =============================
default_args = {
    "owner": "Ismail",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 11),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),

    # =============================
    # EMAIL 1 (Failure / Retry)
    # =============================
    "email": ["ismailismailtj@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
}

# =============================
# DAG Definition
# =============================
dag = DAG(
    "naira_classification_training_daily_test",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
)

# =============================
# Python function to run training
# =============================
def train_naira_model(**kwargs):
    # MLflow setup
    TRACKING_URL = "http://mlflow_service:5000"
    EXPERIMENT_NAME = "NAIRA"

    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable autologging
    mlflow.tensorflow.autolog(log_models=True)

    # Dataset paths
    train_dir = "/opt/airflow/data/Train"
    val_dir = "/opt/airflow/data/val"
    test_dir = "/opt/airflow/data/test"

    # Data generators
    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2
    )

    data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_dataset = data_aug.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32
    )

    val_dataset = data.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32
    )

    # Model definition
    img_size = (224, 224)
    img_shape = img_size + (3,)

    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet"
    )
    mobilenet.trainable = False

    model = Sequential([
        mobilenet,
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(8, activation="softmax")
    ])

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =============================
    # MLflow-tracked training with signature
    # =============================
    with mlflow.start_run(run_name="mobilenetv2-transfer-learning"):
        # Train the model
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1
        )

        # ===== Create input/output example for signature =====
        image_batch, _ = next(train_dataset)                # take one batch from training set
        image_batch = image_batch.astype(np.float32)        # ensure float32 dtype
        prediction_batch = model(image_batch, training=False)  # forward pass

        # ===== Log model with signature =====
    

# =============================
# Training Task
# =============================
train_task = PythonOperator(
    task_id="train_naira_model_task",
    python_callable=train_naira_model,
    dag=dag
)

# =============================
# EMAIL 2 (Success Email)
# =============================
email_task = EmailOperator(
    task_id="send_training_success_email",
    to="ismailismailtj@gmail.com",  
    subject="Naira Classification Training Completed ✅",
    html_content="""
    <h3>Training Completed Successfully</h3>
    <p>The daily Naira classification model training has finished successfully.</p>
    <p><b>Model:</b> MobileNetV2 (Transfer Learning)</p>
    <p><b>Tracking:</b> MLflow</p>
    <p>Timestamp: {{ ds }}</p>
    """,
    dag=dag,
)

# =============================
# Task Dependency
# =============================
train_task >> email_task
