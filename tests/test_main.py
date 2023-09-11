from fastapi.testclient import TestClient
from main import app


def test_index():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == 'Welcome to the Personal Income Prediction Application'

def test_negative():
    data = {
            "workclass": "State-gov",
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "native_country": "United-states",
            "age": 39,
            "fnlwgt": 77516,
            "education_num": 13,
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40
            }
    with TestClient(app) as client:
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        assert response.text == 'Predicted income: <=50K'

def test_positive():
    data = {
            "workclass": "State-gov",
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "native_country": "United-states",
            "age": 39,
            "fnlwgt": 77516,
            "education_num": 13,
            "capital_gain": 10000,
            "capital_loss": 0,
            "hours_per_week": 40
            }
    with TestClient(app) as client:
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        assert response.text == 'Predicted income: >50K'


def test_invalid():
    data = {}
    with TestClient(app) as client:
        response = client.post("/predict", json=data)
        assert response.status_code == 422