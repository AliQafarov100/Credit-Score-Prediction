import os 
import gradio as gr
import requests

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL","http://127.0.0.1:8000")

def predict_score_ui(
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents,
        total_number_of_lates,
        log_transformed_montly_income
):
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": NumberOfDependents,
        "total-number_of_lates": total_number_of_lates,
        "log_transformed-montly-income": log_transformed_montly_income
    }

    response = requests.post(f"{FASTAPI_BASE_URL}/predict/scoring", json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result.get("prediction", "No prediction returned")


with gr.Blocks(title="Credit Score ML Prediction UI") as demo:
    gr.Markdown("# Credit Score ML Prediction UI")
    gr.Markdown("Use this interface to test Credit Scoring prediction models through FastAPI.")


    with gr.Tab("Score Prediction"):
        with gr.Row():
            with gr.Column():
                score_RevolvingUtilizationOfUnsecuredLines = gr.Number(label="RevolvingUtilizationOfUnsecuredLines", value=0.73)
                score_age = gr.Number(label="Age", value=54)
                score_NumberOfTime30_59DaysPastDueNotWorse = gr.Number(label="NumberOfTime30_59DaysPastDueNotWorse", value=4)
                score_DebtRatio = gr.Number(label="DebtRatio", value=0.3)
                score_MonthlyIncome = gr.Number(label="MonthlyIncome", value=4500)
                score_NumberOfOpenCreditLinesAndLoans = gr.Number(label="NumberOfOpenCreditLinesAndLoans", value=3)
                score_NumberOfTimes90DaysLate = gr.Number(label="NumberOfTimes90DaysLate", value=5)
                score_NumberRealEstateLoansOrLines = gr.Number(label="NumberRealEstateLoansOrLines", value=2)
                score_NumberOfTime60_89DaysPastDueNotWorse = gr.Number(label="NumberOfTime60_89DaysPastDueNotWorse", value=1)
                score_NumberOfDependents = gr.Number(label="NumberOfDependents", value=3)
                score_total_number_of_lates = gr.Number(label="total_number_of_lates", value=10)
                score_log_transformed_montly_income = gr.Number(label="log_transformed_montly_income", value=0.6)

                score_button = gr.Button("Predict Score")

            with gr.Column():
                score_output = gr.Textbox(label="Score Prediction Result")

        score_button.click(
            fn=predict_score_ui,
            inputs=[
                score_RevolvingUtilizationOfUnsecuredLines,
                score_age,
                score_NumberOfTime30_59DaysPastDueNotWorse,
                score_DebtRatio,
                score_MonthlyIncome,
                score_NumberOfOpenCreditLinesAndLoans,
                score_NumberOfTimes90DaysLate,
                score_NumberRealEstateLoansOrLines,
                score_NumberOfTime60_89DaysPastDueNotWorse,
                score_NumberOfDependents,
                score_total_number_of_lates,
                score_log_transformed_montly_income,
            ],
            outputs=score_output
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)