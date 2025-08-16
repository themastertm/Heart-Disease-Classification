from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the best model only
model = joblib.load("models/best_model.pkl")

# Define feature order (to keep consistency)
FEATURES = [
    "sex", "age", "cp", "resting_BP", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
    "Max Heart Rate Reserve", "Heart Disease Risk Score"
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Collect features in correct order
        features = [data.get(f) for f in FEATURES]

        if None in features:
            return jsonify({'error': 'Missing one or more input features'}), 400

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        result = {
            'prediction': int(prediction[0]),
            'probability': {
                'No Disease': float(probability[0][0]),
                'Disease': float(probability[0][1])
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictui', methods=['POST'])
def predictui():
    try:
        data = request.form
        features = [float(data.get(f)) for f in FEATURES]

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        result = {
            'prediction': int(prediction[0]),
            'probability': {
                'No Disease': float(probability[0][0]),
                'Disease': float(probability[0][1])
            }
        }

        return render_template('results.html', prediction=result,
    inputs=dict(zip(FEATURES, features[0])))  # user input mapped to names)

    except Exception as e:
        return f'ERROR 500: {str(e)}'


if __name__ == '__main__':
    app.run(debug=True)
