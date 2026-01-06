import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import joblib

PROCESSED_DATA_PATH = Path("data/processed")
MODELS_PATH = Path("models")

MODELS_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("Début du GridSearch")

    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_PATH / "y_train.csv").values.ravel()

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    joblib.dump(best_params, MODELS_PATH / "best_params.pkl")

    print("GridSearch terminé")
    print("Meilleurs paramètres :", best_params)

if __name__ == "__main__":
    main()