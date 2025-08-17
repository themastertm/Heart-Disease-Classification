# src/train.py
from src.utils import load_data, preprocess_data
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_models():
    df = load_data()
    # save_scaler=True so predict can reuse exactly the same scaler
    X_train, X_test, y_train, y_test = preprocess_data(df, save_scaler=True)

    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'ada_boost': AdaBoostClassifier(random_state=42),
        'extra_trees': ExtraTreesClassifier(random_state=42),
        'bagging': BaggingClassifier(random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'xgboost': XGBClassifier(random_state=42, eval_metric="logloss"),
        'lightgbm': LGBMClassifier(random_state=42, verbose=-1),
        'catboost': CatBoostClassifier(random_state=42, verbose=False)
    }

    os.makedirs("../models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        path = f"../models/{name}.pkl"
        joblib.dump(model, path)
        print(f"Saved {name} -> {path}")

if __name__ == "__main__":
    train_models()
