from mlflow import MlflowClient

from config import *
from utils import *


def fail(client, version, reason):
    client.set_model_version_tag(MODEL_NAME, version, "validation_status", "failed")
    client.set_model_version_tag(MODEL_NAME, version, "reason", reason)


def success(client, version):
    client.set_model_version_tag(MODEL_NAME, version, "validation_status", "passed")


def promote(client, version):
    client.set_registered_model_alias(MODEL_NAME, "champion", version)


def main():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Get candidate
    candidate_mv = get_latest_model_version(client, MODEL_NAME)
    version = str(candidate_mv.version)
 

    _, run = get_run_for_version(client, MODEL_NAME, version)

    f1 = get_metric(run, "f1")

    print(f"Candidate v{version} → f1={f1}")

    # 2. Validate metrics exist
    if f1 is None :
        fail(client, version, "missing_metrics")
        return
    
    # 3. Threshold validation
    if f1 < MIN_F1:
        fail(client, version, "low_f1")
        return

    # 4. Compare with champion
    champion = get_champion(client, MODEL_NAME)

    if champion and str(champion.version) == version:
        print(f"Version {version} is already champion")
        return
    
    if champion:
        _, champ_run = get_run_for_version(client, MODEL_NAME, champion.version)
        champ_f1 = get_metric(champ_run, "f1")

        print(f"Champion v{champion.version} → f1={champ_f1}")

        if REQUIRE_BEATS_CHAMPION and champ_f1 is not None and f1 < champ_f1:
            fail(client, version, "worse_than_champion")
            return

    # 5. PASS → promote
    success(client, version)
    promote(client, version)

    print(f"✅ Version {version} promoted to champion")


if __name__ == "__main__":
    main()