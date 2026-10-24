import json
from pathlib import Path


def test_feature_schema_exist():
    schema_path = Path("outputs/feature_schema.json")
    assert schema_path.exists(), "Feature schema file does not exist at"

def test_feature_schema_valid():
    schema_path = Path("outputs/feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Feature schema is not a valid JSON object"


def test_feature_schema_has_content():
    schema_path = Path("outputs/feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "feature_schema.json is empty"