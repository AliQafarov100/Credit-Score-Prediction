import mlflow
from mlflow import MlflowClient
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

SCORE_MODEL_PATH = MODELS_DIR / "score_model_complete_pipeline.joblib"

def load_score_model():
    if not SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Risk model file not found at: {SCORE_MODEL_PATH}"
        )

    model = joblib.load(SCORE_MODEL_PATH)
    model_name = "CreditScoreXGBModel"
    version = "local_file"

    # client = MlflowClient()
    # versions = client.search_model_versions(
    #     "name='CreditScoreXGBModel'"
    # )

    # if not versions:
    #     raise RuntimeError(
    #         "CreditScoreXGBModel is not registered in MLflow"
    #     )

    # latest_version = max(
    #     versions,
    #     key=lambda v: int(v.version)
    # )

    return model, model_name, version

    