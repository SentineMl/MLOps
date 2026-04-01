from dotenv import load_dotenv
import os

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "FraudDetectionModel_Champion")

# Validation thresholds
MIN_F1 = float(os.getenv("MIN_F1", "0.4"))
 
# Behavior
REQUIRE_BEATS_CHAMPION = os.getenv("REQUIRE_BEATS_CHAMPION", "true").lower() == "true"