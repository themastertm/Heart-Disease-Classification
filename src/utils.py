
# src/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

FEATURE_COLS = [
    "sex", "age", "cp", "resting_BP", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",'Max Heart Rate Reserve', 'Heart Disease Risk Score'
]
TARGET_COL = "target"

def load_data():
    df = pd.read_csv("../data/raw/heartv1.csv")
    return df

def preprocess_data(df, save_scaler=False, scaler_path="../models/scaler.pkl", encoder_path="../models/labelencoder.pkl"):
   
    le = LabelEncoder()
    
    df["sex"] = le.fit_transform(df["sex"].astype(str))
    # select columns, ensure order
    df = df.copy()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_scaler:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        joblib.dump(le, encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print('yay')
    return X_train, X_test, y_train, y_test


