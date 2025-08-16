# src/model_selection.py
import joblib, os
from src.utils import load_data, preprocess_data
from sklearn.metrics import accuracy_score

def select_best_model(models_dir="models"):
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df, save_scaler=False)

    best_acc = -1
    best_path = None

    for fn in os.listdir(models_dir):
        if fn.endswith(".pkl") and fn != "best_model.pkl":
            path = os.path.join(models_dir, fn)
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{fn}: acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_path = path

    if best_path:
        joblib.dump(joblib.load(best_path), os.path.join(models_dir, "best_model.pkl"))
        print(f"Best model saved: {best_path} -> models/best_model.pkl (acc={best_acc:.4f})")
        return best_path, best_acc

if __name__ == "__main__":
    select_best_model()
