# src/predict.py
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from src.utils import FEATURE_COLS

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        pass
    return model, scaler

def prepare_and_predict(input_df: pd.DataFrame):
    """
    input_df: pandas DataFrame with columns matching FEATURE_COLS (order not required).
    """
    model, scaler = load_artifacts()
    # ensure columns order and types
    X = input_df[FEATURE_COLS].astype(float).values
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    return preds, proba

if __name__ == "__main__":
    # quick CLI test
    sample = pd.DataFrame([{
        "sex": 1, "age": 45, "cp": 2, "resting_BP": 120, "chol": 200,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2,'Max Heart Rate Reserve':10,'Heart Disease Risk Score': 5
    }])
    print(prepare_and_predict(sample))
