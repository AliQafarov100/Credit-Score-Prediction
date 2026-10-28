import argparse
import mlflow
import mlflow.sklearn

from mlflow import register_model
from mlflow.tracking import MlflowClient

from src.config import MODEL_CONFIG
from src.utils import (
    load_model_table,
    train_test_split,
    save_feature_schema_full,
    save_local_model,
    hash_input,
    write_prediction_log
)

from src.evaluate import evaluate_model, is_eligible_for_production
from src.train import get_preprocessor, build_model_pipeline, get_input_feature_columns


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")
mlflow.set_experiment("credit_score_prediction")

def run_pipeline(model_type: str):
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Invalid model type: {model_type}")

    config = MODEL_CONFIG[model_type]

    print("=" * 70)
    print(f"Starting training pipeline for model: {model_type}")
    print("=" * 70)

    # Step 1) Load the model_table
    df = load_model_table()
    print(f"model_table loaded. Shape: {df.shape}")

    # Step 2) Train/test split
    train_df, test_df = train_test_split(df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Step 3) Identify raw input feature columns
    input_feature_columns = get_input_feature_columns(config)
    print(f"Input feature columns is ready! {input_feature_columns}")

    # Step 4) Build X/y from the dataframe
    X_train = train_df[input_feature_columns].copy()
    y_train = train_df[config["target_column"]].copy()

    X_test = test_df[input_feature_columns].copy()
    y_test = test_df[config["target_column"]].copy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Step 5 build preprocessing + model pipeline
    pipeline = build_model_pipeline(config)
    print("Preprocessing and model pipeline is ready!")

    # Step 6 Train pipeline
    pipeline.fit(X_train, y_train)
    print("Model training complete!")

    # Step 7 Save trained pipeline
    model_path = save_local_model(pipeline, config["local_model_file"])
    print("Local model saved at:", model_path)

    # Step 8 Save feature_schema
    schema_path = save_feature_schema_full(MODEL_CONFIG)
    print("Feature schema saved at:", schema_path)

    # Step 9 Predict on test data
    predictions = pipeline.predict_proba(X_test)[:, 1]

    # Step 10 Evaluate the model
    metrics = evaluate_model(y_test, predictions)
    print("Evaluation model completed!")
    print(f"ROC-AUC Score:           {metrics["ROC-AUC"]:.4f}")
    print(f"Average Precision score: {metrics["Average Precision"]:.4f}")

    # Log full pipeline
    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.log_params(config['params'])
        mlflow.log_param("target_params", config["target_column"])
        mlflow.log_param("input_feature_count", len(input_feature_columns))
        mlflow.log_metric("roc-auc", metrics['ROC-AUC'])
        mlflow.log_metric("average_precision", metrics["Average Precision"])

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

        run_id = run.info.run_id
        print("Pipeline logged to MLflow!")
        print(f"RUN ID: {run_id}")

    # Step 12 Register the model
    result = register_model(
        model_uri=model_info.model_uri, 
        name=config["registered_model_name"]
    )
    model_version = result.version
    print(f"Registered model: {result.name}")
    print(f"Registered version: {model_version}")

    # Step 13 Create MLflow Client for move to staging
    client = MlflowClient()

    client.transition_model_version_stage(
        name=config["registered_model_name"], 
        version=model_version, 
        stage="Staging"
    )
    print(
        f"Model {config['registered_model_name']} "
        f"version {model_version} moved to Staging ✓"
    )

    # Step 14: Production eligibility check
    eligible = is_eligible_for_production(
        roc_auc=metrics["ROC-AUC"],
        average_precision=metrics["Average Precision"],
        roc_auc_threshold=config["promotion_roc_auc_threshold"],
        average_precision_threshold=config["promotion_average_precision"]
    )

    if eligible:
        # Step 15: Move to production
        client.transition_model_version_stage(
            name=config["registered_model_name"],
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(
            f"Model {config['registered_model_name']} "
            f"version {model_version} moved to Production ✓"
        )

        # Step 16: Load production pipeline
        production_model = mlflow.sklearn.load_model(
            model_uri=f"models:/{config['registered_model_name']}/Production"
        )

        # Step 17: Sample prediction from production pipeline
        sample_df = X_test.head(5).copy()
        sample_predictions = production_model.predict_proba(sample_df)[:, 1]
        print("Sample Predictions:", sample_predictions)

        # Step 18: Prediction logging
        input_payload = sample_df.head(1).to_dict(orient="records")[0]
        input_hash = hash_input(input_payload)

        log_file = write_prediction_log(
            model_name=config["registered_model_name"],
            model_version=str(model_version),
            input_hash=input_hash,
            prediction=str(sample_predictions[0])
        )

        print("Prediction logged ✓")
        print("Log file location:", log_file)
    else:
        print("Model is NOT eligible for Production")

if __name__ == "__main__":
    run_pipeline("score")
