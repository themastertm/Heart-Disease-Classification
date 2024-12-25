from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


models = {
    'Logistic Regression': joblib.load('Deployed Models/Logistic Regression.joblib'),
    'Random Forest': joblib.load('Deployed Models/Random Forest.joblib'),
    'SVM': joblib.load('Deployed Models/SVM.joblib'),
    'Ada Boost': joblib.load('Deployed Models/Ada Boost.joblib'),
    'Bagging': joblib.load('Deployed Models/Bagging.joblib'),
    'CatBoost': joblib.load('Deployed Models/CatBoost.joblib'),
    'LightGBM': joblib.load('Deployed Models/LightGBM.joblib'),
    'XGBoost': joblib.load('Deployed Models/XGBoost.joblib'),
    'Gradient Boosting': joblib.load('Deployed Models/Gradient Boosting.joblib'),
    'Extra Trees': joblib.load('Deployed Models/Extra Trees.joblib')
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = [
            data.get('sex'),
            data.get('age'),
            data.get('cp'),
            data.get('resting_BP'),
            data.get('chol'),
            data.get('fbs'),
            data.get('restecg'),
            data.get('thalach'),
            data.get('exang'),
            data.get('oldpeak'),
            data.get('slope'),
            data.get('ca'),
            data.get('thal'),
            data.get('Max Heart Rate Reserve'),
            data.get('Heart Disease Risk Score')
        ]

        if None in features:
            return jsonify({'error': 'Missing one or more input features'}), 400

        features = np.array(features).reshape(1, -1)

        # Collect predictions from all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(features)
            probability = model.predict_proba(features)
            predictions[model_name] = {
                'prediction': int(prediction[0]),
                'probability': {
                    'No Disease': float(probability[0][0]),
                    'Disease': float(probability[0][1])
                }
            }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictui', methods=['POST'])
def predictui():
    try:
        data = request.form

        features = [
            float(data.get('sex')),
            float(data.get('age')),
            float(data.get('cp')),
            float(data.get('resting_BP')),
            float(data.get('chol')),
            float(data.get('fbs')),
            float(data.get('restecg')),
            float(data.get('thalach')),
            float(data.get('exang')),
            float(data.get('oldpeak')),
            float(data.get('slope')),
            float(data.get('ca')),
            float(data.get('thal')),
            float(data.get('Max Heart Rate Reserve')),
            float(data.get('Heart Disease Risk Score'))
        ]

        if None in features:
            return 'ERROR 400, missing one or more input features'

        features = np.array(features).reshape(1, -1)

        # Collect predictions from all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(features)
            probability = model.predict_proba(features)
            predictions[model_name] = {
                'prediction': int(prediction[0]),
                'probability': {
                    'No_Disease': float(probability[0][0]),
                    'Disease': float(probability[0][1])
                }
            }

        return render_template('results.html', predictions=predictions)

    except Exception as e:
        return 'ERROR 500: {}'.format(str(e))

if __name__ == '__main__':
    app.run(debug=True)
