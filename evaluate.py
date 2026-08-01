"""Evaluate the trained SafeChat-AI model on the held-out test split
produced by train.py.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cyberbullying_model.joblib"
TEST_PATH = BASE_DIR / "processed" / "test.csv"


def evaluate() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run `python train.py` first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"No test split found at {TEST_PATH}. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text"])

    predictions = model.predict(test_df["text"])

    acc = accuracy_score(test_df["label"], predictions)
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(test_df["label"], predictions, target_names=["safe", "bullying"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(test_df["label"], predictions))


if __name__ == "__main__":
    evaluate()
