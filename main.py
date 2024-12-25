# Client code to send a request and print predictions
import requests

url = "http://127.0.0.1:5000/predict"

data = [float(x) for x in input("Input all features separated with a comma:\n").split(',')]

payload = {
    "sex": data[0],
    "age": data[1],
    "cp": data[2],
    "resting_BP": data[3],
    "chol": data[4],
    "fbs": data[5],
    "restecg": data[6],
    "thalach": data[7]  ,
    "exang": data[8],
    "oldpeak": data[9],
    "slope": data[10],
    "ca": data[11],
    "thal": data[12],
    "Max Heart Rate Reserve": data[13],
    "Heart Disease Risk Score": data[14]
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Predictions from all models:")
    for model in response.json():
        print('-' * 100)
        print(f'Model Name: {model}\nModel Prediction: {response.json()[model]["prediction"]}\nModel Probabilities: {response.json()[model]["probability"]}')
else:
    print(f"Error {response.status_code}: {response.json()}")
