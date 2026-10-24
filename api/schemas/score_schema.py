from pydantic import BaseModel, Field

class ScorePredictRequest(BaseModel):
    
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., examples=[0.65])
    age: int = Field(..., examples=[32])
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse", examples=[2]
    )
    DebtRatio: float = Field(..., examples=[0.3])
    MonthlyIncome: float = Field(..., examples=[5000.0])
    NumberOfOpenCreditLinesAndLoans: int = Field(..., examples=[4])
    NumberOfTimes90DaysLate: int = Field(..., examples=[6])
    NumberRealEstateLoansOrLines: int = Field(..., examples=[2])
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse", examples=[10]
    )
    NumberOfDependents: int = Field(..., examples=[3])
    total_number_of_lates: int = Field(
        ..., alias="total-number_of_lates", examples=[4]
    )
    log_transformed_monthly_income: float = Field(
        ..., alias="log_transformed-montly-income", examples=[8.52]
    )