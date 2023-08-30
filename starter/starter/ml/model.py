from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn import svm

from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = svm.SVC()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metric_on_slice(test_df, fixed_feature, cat_features, model, encoder, lb):
    """
    Function for calculating model metrics when the value of a given feature is held fixed.

    :param test_df: Testing data
    :type test_df: pd.Dataframe
    :param fixed_feature: The feature to be fixed when calculating model metrics
    :type fixed_feature: str
    :param cat_features: Category features
    :type cat_features: list
    :param model: The model to run inference
    :param encoder: Trained OneHotEncoder if training is True, otherwise returns the encoder 
    passed in.
    :type encoder: sklearn.preprocessing._encoders.OneHotEncoder
    :param lb: Trained LabelBinarizer if training is True, otherwise returns the binarizer 
    passed in.
    :type lb: sklearn.preprocessing._label.LabelBinarizer
    :return: Result of metrics with each value
    :rtype: dict
    """
    result = {}
    for value in test_df[fixed_feature].unique():
        slice_df = test_df[test_df[fixed_feature] == value]
        X_slice, y_slice, _, _  = process_data(slice_df, cat_features, 
                                               label="salary", training=False, 
                                               encoder=encoder, lb=lb)
        
        pred_slice = inference(model, X_slice)

        precision, recall, fbeta = compute_model_metrics(y_slice, pred_slice)
        result[value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        }
    return result


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : SVM Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def export_slice_metric(metric_rslt):
    """
    Export the result of model metric on data slice into txt file

    :param metric_rslt: Result of metrics with each value
    :type metric_rslt: dict
    """
    with open('slice_output.txt', 'w') as f:
        for key, value in metric_rslt.items():
            f.write(f"{key}: {value}")
            f.write("\n")
    print("[Model] Exported the slice metric to slice_output.txt")