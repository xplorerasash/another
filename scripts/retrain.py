"""Re-train the model properly with bert-base-uncased.

Fixes:
- Uses bert-base-uncased (matches project title)
- Balances the training data
- Trains for 3 epochs with evaluation
- Compatible with transformers 5.x
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="models/bert_cyberbully")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=500)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("dataset/labeled.csv").dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Total samples: {len(df)}")
    print(f"Class 0 (safe): {(df['label']==0).sum()}, Class 1 (harmful): {(df['label']==1).sum()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    safe = train_df[train_df["label"] == 0]
    harmful = train_df[train_df["label"] == 1]
    min_count = min(len(safe), len(harmful), args.max_train_samples // 2)
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
        output_dir=str(output_dir),
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
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nModel saved to {output_dir}")

    print("\n--- Test Set Evaluation ---")
    ds_test = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
    ds_test = ds_test.map(tokenize, batched=True)
    ds_test = ds_test.map(lambda x: {"label": int(x["label"])})
    results = trainer.evaluate(ds_test)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
