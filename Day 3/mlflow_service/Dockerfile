# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /mlflow

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Expose MLflow port
EXPOSE 5000

# Default command to run MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "/mlflow/mlruns", "--default-artifact-root", "/mlflow/mlruns", "--host", "0.0.0.0", "--port", "5000"]

