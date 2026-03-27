import os
import mlflow
from config import *

def setup_mlflow(experiment_name):
    # Set the environment variables for S3 here so train_eval stays clean!
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return mlflow

def log_metrics(mlflow, metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model(mlflow, model, model_name, artifact_path="model"):
    mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=model_name
        )
    print("MLflow logging complete!")
