from fastapi import APIRouter
from monitoring.drift_monitor import run_psi_monitor

router = APIRouter()

@router.get("/psi")
def psi_monitor():
    result = run_psi_monitor()
    return result