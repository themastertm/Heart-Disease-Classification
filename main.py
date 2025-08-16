import requests

url = "http://127.0.0.1:5000/predict"

FEATURES = [
    "sex", "age", "cp", "resting_BP", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
    "Max Heart Rate Reserve", "Heart Disease Risk Score"
]

# Ask user for input
print(f"Enter values for the following features in order:\n{', '.join(FEATURES)}")
data = [float(x) for x in input("Input all 15 features separated by commas:\n").split(',')]

payload = {FEATURES[i]: data[i] for i in range(len(FEATURES))}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Best Model Prediction:")
    print(f"Prediction: {response.json()['prediction']}")
    print(f"Probabilities: {response.json()['probability']}")
else:
    print(f"Error {response.status_code}: {response.json()}")
