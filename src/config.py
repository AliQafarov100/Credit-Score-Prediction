MODEL_CONFIG = {
    "score": {
        "run_name": "XGBoost-Score",
        "registered_model_name": "CreditScoreXGBModel",

        # Core columns
        "target_column": "SeriousDlqin2yrs",

        # Input Features
        "input_features": [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
            "total-number_of_lates",
            "log_transformed-montly-income"
        ],

        # Feature types
        "numeric_features": [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
            "total-number_of_lates",
            "log_transformed-montly-income"
        ],

        # Model thresholds
        "promotion_roc_auc_threshold": 0.75,
        "promotion_average_precision": 0.25,

        # Local Storage
        "local_model_file": "score_model_complete_pipeline.joblib",

        # Model parameters
        "params": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.014831660073622589,
            "subsample": 0.815584046804639,
            "colsample_bytree": 0.763165386967939,
            "reg_alpha": 8.750244277935877,
            "reg_lambda": 7.195684092208847,
            "scale_pos_weight": 1.8630641212756873
        }
    }
}