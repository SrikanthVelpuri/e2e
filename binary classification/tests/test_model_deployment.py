import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from binary_classification.model_deployment import app, PredictionInput

client = TestClient(app)

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.pyfunc.load_model') as mock_load_model:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_model.predict.return_value = [1]
        mock_load_model.return_value = mock_model
        yield mock_load_model

def test_predict_endpoint(mock_mlflow):
    response = client.post("/predict", json={"features": {"age": 25, "workclass": 1, "education": 2, "marital-status": 1, "occupation": 3, "relationship": 1, "race": 1, "sex": 1, "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": 1}})
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "probability": 0.8,
        "model_version": mock_mlflow.return_value.version
    }

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "1"}
