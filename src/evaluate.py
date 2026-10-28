from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(y_test, predictions) -> dict: 
    roc_auc = roc_auc_score(y_test, predictions)
    average_precsion = average_precision_score(y_test, predictions)

    return {
        "ROC-AUC": roc_auc,
        "Average Precision": average_precsion,
    }

# This function checks if the model meets the specified thresholds for roc-auc and ap to be eligible
# for production promotion
def is_eligible_for_production(
        roc_auc: float, 
        average_precision: float, 
        roc_auc_threshold: float, 
        average_precision_threshold: float
    ) -> bool:
    return{
        roc_auc >= roc_auc_threshold and
        average_precision >= average_precision_threshold
    }