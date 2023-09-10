import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from pydantic import BaseModel

from ml.model import load_model, inference
from ml.data import process_data


app = FastAPI()

CATEGORY_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

model, encoder, lb = load_model('model')


class Data(BaseModel):
    workclass: str = None
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.get("/")
def read_root():
    response = Response(
        status_code=status.HTTP_200_OK,
        content="Welcome to the Personal Income Prediction Application"
    )
    return response


@app.post("/predict")
def predict(data: Data):
    data = pd.DataFrame([data.dict()])
    print(encoder)
    X, _, _, _ = process_data(data,
                              categorical_features=CATEGORY_FEATURES,
                              label=None, training=False,
                              encoder=encoder, lb=lb)
    y_pred = inference(model, X)
    label_pred = lb.inverse_transform(y_pred)[0]
    response = Response(
        status_code=status.HTTP_200_OK,
        content=f"Predicted income: {label_pred}",
    )

    return response
