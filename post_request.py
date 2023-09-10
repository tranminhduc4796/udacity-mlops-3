import requests

# Post response from the server
sample_dict = {
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
url = "https://udacity-mlops-3.onrender.com/predict"
post_response = requests.post(url, json=sample_dict)
print(post_response.status_code)
print(post_response.content)
