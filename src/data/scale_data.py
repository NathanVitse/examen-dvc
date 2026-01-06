import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("models")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("Début de la normalisation")

    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_PATH / "X_test.csv")

    X_train = X_train.select_dtypes(include=["float64", "int64"])
    X_test = X_test.select_dtypes(include=["float64", "int64"])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(PROCESSED_DATA_PATH / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DATA_PATH / "X_test_scaled.csv", index=False)

    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")

    print("Normalisation terminée avec succès")

if __name__ == "__main__":
    main()
