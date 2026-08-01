"""Evaluate the trained SafeChat-AI model on the held-out test split
produced by train.py.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cyberbullying_model.joblib"
TEST_PATH = BASE_DIR / "processed" / "test.csv"
REPORT_PATH = BASE_DIR / "models" / "eval_report.json"


def evaluate() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run `python train.py` first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"No test split found at {TEST_PATH}. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text"])

    predictions = model.predict(test_df["text"])
    probabilities = model.predict_proba(test_df["text"])[:, 1]

    acc = accuracy_score(test_df["label"], predictions)
    p, r, f1, _ = precision_recall_fscore_support(test_df["label"], predictions, average="binary")
    auc = roc_auc_score(test_df["label"], probabilities)
    cm = confusion_matrix(test_df["label"], predictions)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}\n")
    print(classification_report(test_df["label"], predictions, target_names=["safe", "harmful"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    report = {
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn_safe_correct": int(cm[0][0]),
            "fp_safe_misclassified": int(cm[0][1]),
            "fn_harmful_missed": int(cm[1][0]),
            "tp_harmful_correct": int(cm[1][1]),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
