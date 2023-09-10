import pytest

import pandas as pd
from ml.model import load_model, inference, compute_model_metrics, compute_model_metric_on_slice
from ml.data import process_data

@pytest.fixture
def data():
    data = pd.read_csv("data/census.csv")
    return data

@pytest.fixture
def model_comps():
    return load_model()

@pytest.fixture
def processed_data(data, model_comps):

    _, enc, lb = model_comps
    
    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    ]
    X, y, _, _  = process_data(data, cat_features, 
                                               label="salary", training=False, 
                                               encoder=enc, lb=lb)
    return X, y



def test_data_shape(data):
    assert data.shape == data.dropna().shape


def test_inference(model_comps, processed_data):
    model, _, _ = model_comps
    X, y = processed_data
    preds = inference(model, X)
    assert preds.shape == y.shape

def test_model_metric(model_comps, processed_data):
    model, _, _ = model_comps
    X, y = processed_data
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision >= 0 and precision <= 1
    assert recall >= 0 and recall <= 1
    assert fbeta >=0 and fbeta <= 1
    