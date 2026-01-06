import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib

# Chemins
PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("models")

MODELS_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("Début de l'entraînement du modèle")

    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_PATH / "y_train.csv").values.ravel()

    best_params = joblib.load(MODELS_PATH / "best_params.pkl")

    model = RandomForestRegressor(
        **best_params,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODELS_PATH / "model.pkl")

    print("Modèle entraîné et sauvegardé avec succès")

if __name__ == "__main__":
    main()