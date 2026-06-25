from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "model_unavailable"]
    assert data["model"] == "LSTM RUL Regressor"
    assert isinstance(data["model_loaded"], bool)


def test_predict_bad_shape():
    response = client.post("/predict", json={"sequence": [[0.0] * 24] * 10})
    assert response.status_code == 400 or response.status_code == 503
    if response.status_code == 400:
        assert "Inference window expects matrix shape" in response.json()["detail"]
    else:
        assert response.json()["detail"] == "Model weights are not loaded. Please ensure models/lstm_weights.pth exists."
