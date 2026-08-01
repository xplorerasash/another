"""Fine-tune a Hugging Face transformer on the provided labeled dataset.

This script prepares reproducible train/validation/test splits (70/15/15),
saves them to `processed/` and uses the train/validation splits for Trainer.

Usage:
  python scripts/train_hf.py --data_csv dataset/labeled.csv --model_name_or_path xlm-roberta-base --output_dir models/hf_cyberbully
  python scripts/train_hf.py --balance_train --epochs 3 --batch_size 8
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, default="dataset/labeled.csv")
    p.add_argument("--model_name_or_path", type=str, default="xlm-roberta-base")
    p.add_argument("--output_dir", type=str, default="models/hf_cyberbully")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0, help="Optional cap on samples for quick experiments (0=all)")
    p.add_argument("--balance_train", action='store_true', help="Undersample majority class in training set to match minority class count")
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    data_path = Path(args.data_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df = df.dropna(subset=['text', 'label'])
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)

    # Ensure both classes present
    labels = sorted(df['label'].unique().tolist())
    if not (0 in labels and 1 in labels):
        raise ValueError(f"Dataset must contain both classes 0 and 1. Found labels: {labels}")

    # Create 70/15/15 splits reproducibly (stratified)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Optionally balance the training set by undersampling the majority class
    if args.balance_train:
        safe = train_df[train_df['label'] == 0]
        harmful = train_df[train_df['label'] == 1]
        min_count = min(len(safe), len(harmful))
        safe_sampled = safe.sample(n=min_count, random_state=42)
        harmful_sampled = harmful.sample(n=min_count, random_state=42)
        train_df = pd.concat([safe_sampled, harmful_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced training set: {len(train_df)} samples (safe={min_count}, harmful={min_count})")

    processed_dir = Path(__file__).resolve().parent.parent / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_csv = processed_dir / 'train.csv'
    val_csv = processed_dir / 'validation.csv'
    test_csv = processed_dir / 'test.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved splits: {train_csv} ({len(train_df)}), {val_csv} ({len(val_df)}), {test_csv} ({len(test_df)})")

    # Tokenize and prepare HF Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))

    ds_train = ds_train.map(tokenize_fn, batched=True)
    ds_val = ds_val.map(tokenize_fn, batched=True)

    # ensure label column exists and is int
    if 'label' in ds_train.column_names:
        ds_train = ds_train.map(lambda x: {'label': int(x['label'])})
        ds_val = ds_val.map(lambda x: {'label': int(x['label'])})

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

    # Build TrainingArguments compatibly: some transformers versions do not
    # accept newer kwargs like `evaluation_strategy`/`save_strategy`. Inspect
    # the constructor and pass only supported parameters so the script works
    # across a wider range of transformers releases.
    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs = {
        'output_dir': args.output_dir,
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'logging_steps': 50,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'f1',
    }

    # Conditionally include newer options when supported by the installed transformers
    has_eval = 'evaluation_strategy' in ta_sig.parameters
    has_save = 'save_strategy' in ta_sig.parameters
    if has_eval:
        ta_kwargs['evaluation_strategy'] = 'epoch'
    if has_save:
        ta_kwargs['save_strategy'] = 'epoch'

    # Some older transformers may not support evaluation/save strategies
    # or may require them to match when using load_best_model_at_end. If the
    # installed version doesn't support evaluation or the config would be
    # inconsistent, disable `load_best_model_at_end` to avoid runtime errors.
    if not has_eval or not has_save:
        ta_kwargs['load_best_model_at_end'] = False

    training_args = TrainingArguments(**ta_kwargs)

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    # Build Trainer kwargs compatibly across transformers versions
    import inspect

    trainer_init_sig = inspect.signature(Trainer.__init__)
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': ds_train,
        'eval_dataset': ds_val,
        'compute_metrics': compute_metrics,
    }
    if 'tokenizer' in trainer_init_sig.parameters:
        trainer_kwargs['tokenizer'] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
