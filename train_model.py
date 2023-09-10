# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metric_on_slice, export_slice_metric, export_model_files, compute_model_metrics
import pandas as pd

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
test_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
print(f"Precision: {precision}; Recall: {recall}; F-Beta: {fbeta}")
metric_rslt = compute_model_metric_on_slice(test, "marital_status", cat_features, model, encoder, lb)
export_slice_metric(metric_rslt)
export_model_files(model, encoder, lb)