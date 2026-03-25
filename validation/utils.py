from typing import Optional
from mlflow import MlflowClient
from mlflow.exceptions import RestException


def get_metric(run, metric_name: str) -> Optional[float]:
    return run.data.metrics.get(metric_name)


def get_latest_model_version(client: MlflowClient, model_name: str):
    versions = list(client.search_model_versions(f"name = '{model_name}'"))
    if not versions:
        raise RuntimeError(f"No versions found for {model_name}")

    return max(versions, key=lambda v: int(v.version))


def get_run_for_version(client: MlflowClient, model_name: str, version: str):
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    return mv, run


def get_champion(client: MlflowClient, model_name: str):
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except RestException:
        return None