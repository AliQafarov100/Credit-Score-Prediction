from fastapi import FastAPI
from api.routers import scoring, monitoring

app = FastAPI(title="CredictDefault API")

@app.get("/")
def root():
    return {"message": "CreditDefault API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Registering the routers
app.include_router(scoring.router, prefix="/predict", tags=["Credit Score Prediction"])
app.include_router(monitoring.router, prefix="/monitor", tags=["Monitoring"])

