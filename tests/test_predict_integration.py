from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_predict_valid_shape():
    # A valid 30x24 input should either produce a prediction or return 503 if the model is unavailable.
    payload = {"sequence": [[0.0] * 24 for _ in range(30)]}
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_remaining_useful_life" in data
        assert "unit_status" in data
        assert isinstance(data["predicted_remaining_useful_life"], float)
        assert data["unit_status"] in ["Critical Attention Required", "Normal Operational Status"]
    else:
        assert response.json()["detail"] == "Model weights are not loaded. Please ensure models/lstm_weights.pth exists."
