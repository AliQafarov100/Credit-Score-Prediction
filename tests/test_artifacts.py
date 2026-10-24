from pathlib import Path

def test_score_model_exists():
    model_path = Path("models/score_model_complete_pipeline.joblib")
    assert model_path.exists(), "Score model file does not exist at"
