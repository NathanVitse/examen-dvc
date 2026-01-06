import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DATA_PATH = Path("data/raw_data/raw.csv")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW_DATA_PATH)

    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)

    print("Train/test split terminé avec succès.")

if __name__ == "__main__":
    main()