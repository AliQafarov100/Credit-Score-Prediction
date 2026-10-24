from fastapi import APIRouter
from api.schemas.score_schema import ScorePredictRequest
from api.services.predictor import predict_score_result


router = APIRouter()

@router.post("/scoring")
def score_status(request: ScorePredictRequest):
    result = predict_score_result(request.model_dump(by_alias=True))
    return result
