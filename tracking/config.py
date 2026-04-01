import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION=os.getenv("AWS_DEFAULT_REGION")
MLFLOW_S3_ENDPOINT_URL=os.getenv("MLFLOW_S3_ENDPOINT_URL")
TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME=os.getenv("MLFLOW_EXPERIMENT_NAME")

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "sentinel")
DB_USER = os.getenv("DB_USER", "sentinel_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "sentinel_pass")
