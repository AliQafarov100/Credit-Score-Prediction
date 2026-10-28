from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# This function returns the raw input feature columns for the selected model.
def get_input_feature_columns(config: dict) -> list[str]:
    return config["input_features"]

# This function builds preprocessing pipeline dynimacally using numeric features 
# from MODEL_CONFIG
def get_preprocessor(config: dict) -> ColumnTransformer:
    numeric_features = config['numeric_features']

    numeric_transformer = Pipeline(steps=[
        ('imputer',  SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ], 
        remainder='drop'
    )

    return preprocessor


# This function build sklearn pipeline
def build_model_pipeline(config: dict) -> Pipeline:
    preprocessor = get_preprocessor(config)

    model = XGBClassifier(
        n_estimators=config["params"]["n_estimators"],
        max_depth=config["params"]["max_depth"],
        learning_rate=config["params"]["learning_rate"],
        subsample=config["params"]["subsample"],
        colsample_bytree=config["params"]["colsample_bytree"],
        reg_alpha=config["params"]["reg_alpha"],
        reg_lambda=config["params"]["reg_lambda"],
        scale_pos_weight=config["params"]["scale_pos_weight"]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
