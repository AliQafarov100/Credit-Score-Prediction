from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CreditDefault API is running"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_score_route_exists():
    response = client.post("/predict/scoring", json={})
    # Expect validation error since body is required
    assert response.status_code in [200, 422]

def test_monitoring_psi():
    response = client.get("/monitor/psi")
    assert response.status_code == 200