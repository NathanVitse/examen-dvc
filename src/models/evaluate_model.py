import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import joblib
import json
import numpy as np

PROCESSED_DATA_PATH = Path("data/processed")
METRICS_PATH = Path("metrics")
MODELS_PATH = Path("models")

METRICS_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("Début de l'évaluation du modèle")

    model = joblib.load(MODELS_PATH / "model.pkl")
    X_test = pd.read_csv(PROCESSED_DATA_PATH / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_PATH / "y_test.csv").values.ravel()

    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred
    })
    pred_df.to_csv(PROCESSED_DATA_PATH / "predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    with open(METRICS_PATH / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Évaluation terminée")
    print("Scores :", scores)

if __name__ == "__main__":
    main()