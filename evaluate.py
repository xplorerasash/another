"""Evaluate the fine-tuned BERT moderation model on the held-out test split.

The test split is produced by `scripts/retrain.py` (saved to
`processed/balanced_test.csv`) so the numbers here match the report written
during training. Results are written to `models/eval_report.json`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bert_cyberbully"
TEST_PATH = BASE_DIR / "processed" / "balanced_test.csv"
REPORT_PATH = BASE_DIR / "models" / "eval_report.json"
MAX_LENGTH = 64
BATCH_SIZE = 32


def _predict_batched(model, tokenizer, texts, device):
    model.eval()
    all_logits = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, truncation=True, padding="max_length", max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        all_logits.append(logits.cpu().numpy())
    return np.vstack(all_logits)


def evaluate() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No fine-tuned BERT model found at {MODEL_PATH}. Run `python scripts/retrain.py` first."
        )
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"No test split found at {TEST_PATH}. Run `python scripts/retrain.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH)).to(device)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    y_true = test_df["label"].astype(int).to_numpy()

    logits = _predict_batched(model, tokenizer, test_df["text"].tolist(), device)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    y_pred = logits.argmax(-1)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, probs[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Evaluating {str(MODEL_PATH)} on {len(test_df)} test samples ({device})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["safe", "harmful"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    report = {
        "dataset": str(TEST_PATH),
        "samples": {"test": int(len(test_df))},
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "tn_safe_correct": int(tn),
            "fp_safe_misclassified": int(fp),
            "fn_harmful_missed": int(fn),
            "tp_harmful_correct": int(tp),
        },
        "classification_summary": {
            "safe_support": int(tn + fp),
            "harmful_support": int(fn + tp),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
