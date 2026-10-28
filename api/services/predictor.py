import pandas as pd
from api.services.model_loader import load_score_model
from monitoring.logger import log_prediction



def predict_score_result(data: dict):
    model, model_name, model_version = load_score_model()

    # Convert input to dataframe
    input_df = pd.DataFrame([data])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    # Logging
    log_prediction(
        model_name=model_name,
        model_version=model_version,
        input_data=data,
        prediction=prediction,
        probability=probability
    )

    return {
        "prediction": prediction,
        "default_probability": probability,
        "model_name": model_name,
        "model_version": str(model_version),
    }
