"""Re-train the model properly with bert-base-uncased.

Fixes:
- Uses bert-base-uncased (matches project title)
- Trains on the balanced dataset (dataset/balanced.csv) so the model does
  not over-predict "harmful" on normal conversational text
- Trains for 3 epochs with evaluation (f1-based model selection)
- Compatible with transformers 5.x
- Enforces a quality gate: the model is trained into a staging dir and
  only promoted to the final output dir if test-set metrics clear the
  thresholds in model_quality.py. A failed retrain keeps the previous
  model and exits non-zero.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_quality import DEFAULT_THRESHOLDS, check_quality


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="models/bert_cyberbully")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument(
        "--data_csv",
        type=str,
        default="dataset/balanced.csv",
        help="Balanced dataset. Falls back to dataset/labeled.csv if missing.",
    )
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Cap on training samples per class. 0 = use all available.",
    )
    p.add_argument("--eval_report", default="models/eval_report.json")
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    import numpy as np
    preds = logits.argmax(-1)
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exps / exps.sum(axis=-1, keepdims=True)
    probs = probs[:, 1]
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}


def main():
    args = parse_args()
    final_dir = Path(args.output_dir)
    staging_dir = final_dir.parent / (final_dir.name + "_staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    data_csv = Path(args.data_csv)
    if not data_csv.exists() or data_csv.name == "labeled.csv":
        print("Generating balanced dataset from dataset/labeled.csv...")
        subprocess.run([sys.executable, "scripts/build_balanced_dataset.py"], check=True)
        data_csv = Path("dataset/balanced.csv")

    if not data_csv.exists():
        fallback = Path("dataset/labeled.csv")
        if fallback.exists():
            print(f"{data_csv} not found; falling back to {fallback}")
            data_csv = fallback
        else:
            raise FileNotFoundError(f"No dataset found at {args.data_csv} or {fallback}")

    df = pd.read_csv(data_csv).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Total samples: {len(df)}")
    print(f"Class 0 (safe): {(df['label']==0).sum()}, Class 1 (harmful): {(df['label']==1).sum()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    test_path = Path("processed/balanced_test.csv")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df[["text", "label"]].to_csv(test_path, index=False)
    print(f"Saved test split to {test_path}")

    safe = train_df[train_df["label"] == 0]
    harmful = train_df[train_df["label"] == 1]
    min_count = min(len(safe), len(harmful))
    if args.max_train_samples and args.max_train_samples > 0:
        min_count = min(min_count, args.max_train_samples // 2)
    balanced = pd.concat([
        safe.sample(n=min_count, random_state=42),
        harmful.sample(n=min_count, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced train: {len(balanced)} (safe={min_count}, harmful={min_count})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    ds_train = Dataset.from_pandas(balanced[["text", "label"]])
    ds_val = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)
    ds_train = ds_train.map(lambda x: {"label": int(x["label"])})
    ds_val = ds_val.map(lambda x: {"label": int(x["label"])})

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    ta = TrainingArguments(
        output_dir=str(staging_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=50,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(staging_dir))
    tokenizer.save_pretrained(str(staging_dir))

    print(f"\nModel trained in staging dir: {staging_dir}")

    print("\n--- Test Set Evaluation ---")
    ds_test = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
    ds_test = ds_test.map(tokenize, batched=True)
    ds_test = ds_test.map(lambda x: {"label": int(x["label"])})

    import numpy as np
    preds = trainer.predict(ds_test)
    logits = preds.predictions
    y_true = np.asarray(preds.label_ids)
    y_pred = logits.argmax(-1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, probs[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  accuracy:  {acc:.4f}")
    print(f"  precision: {p:.4f}")
    print(f"  recall:    {r:.4f}")
    print(f"  f1:        {f1:.4f}")
    print(f"  roc_auc:   {auc:.4f}")
    print("  confusion matrix (rows=actual [safe, harmful], cols=predicted [safe, harmful]):")
    print(cm)

    report = {
        "dataset": str(data_csv),
        "samples": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
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

    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}
    passed, failures = check_quality(metrics)
    report["quality_gate"] = {
        "passed": bool(passed),
        "minimum_thresholds": dict(DEFAULT_THRESHOLDS),
        "failures": {
            name: {"actual": None if actual is None else round(actual, 4), "minimum": minimum}
            for name, (actual, minimum) in failures.items()
        },
    }

    report_path = Path(args.eval_report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved eval report to {report_path}")

    if passed:
        print("\n[PASS] Model meets all quality thresholds. Accepting model.")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(staging_dir), str(final_dir))
        print(f"Model accepted and saved to {final_dir}")
        return

    print("\n[FAIL] Model rejected. Metrics below minimum thresholds:")
    for name, (actual, minimum) in failures.items():
        actual_str = "missing" if actual is None else f"{actual:.4f}"
        print(f"  - {name}: {actual_str} < {minimum:.4f}")
    print(f"Keeping previously accepted model at {final_dir} (if present).")
    shutil.rmtree(staging_dir, ignore_errors=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
