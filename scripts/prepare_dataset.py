"""Prepare a canonical `text,label` CSV for training.

This script reads common dataset schemas (Kaggle hate/offensive schema,
or CSVs with 'tweet'/'text' columns) and produces `dataset/labeled.csv`
with exactly two columns: 'text' and 'label' where label is 0 (safe)
or 1 (harmful). If the original file is overwritten, a backup is kept as
`dataset/labeled.orig.csv`.
"""
from pathlib import Path
import shutil
import pandas as pd


def map_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    # If already in desired format
    if 'text' in df.columns and 'label' in df.columns:
        return df[['text', 'label']]

    # Common Kaggle schema: 'tweet' and 'class' where class: 0=hate,1=offensive,2=neither
    if 'tweet' in df.columns and 'class' in df.columns:
        df2 = pd.DataFrame()
        df2['text'] = df['tweet'].astype(str)
        df2['label'] = df['class'].apply(lambda c: 1 if int(c) in (0, 1) else 0)
        return df2

    # Other schema: columns 'text' and maybe 'hate_speech'/'offensive_language'
    if 'text' in df.columns:
        df2 = pd.DataFrame()
        df2['text'] = df['text'].astype(str)
        if 'label' in df.columns:
            df2['label'] = df['label'].astype(int)
            return df2
        # heuristics: if hate_speech/offensive_language exist
        if 'hate_speech' in df.columns or 'offensive_language' in df.columns:
            df2['label'] = ((df.get('hate_speech', 0) > 0) | (df.get('offensive_language', 0) > 0)).astype(int)
            return df2

    raise ValueError('Unrecognized dataset schema. Please provide a CSV with columns (tweet,class) or (text,label).')


def main():
    src = Path('dataset') / 'labeled.csv'
    if not src.exists():
        raise FileNotFoundError(f"Expected dataset/labeled.csv but not found at {src}")

    # some pandas versions don't accept the 'errors' parameter
    try:
        df = pd.read_csv(src, dtype=str, encoding='utf-8', errors='replace')
    except TypeError:
        df = pd.read_csv(src, dtype=str, encoding='utf-8')

    out = map_to_binary(df)

    # ensure both classes present
    labels = set(out['label'].astype(int).unique().tolist())
    if not (0 in labels and 1 in labels):
        raise ValueError(f"Prepared dataset does not contain both classes 0 and 1. Found: {labels}")

    backup = src.with_suffix('.orig.csv')
    if not backup.exists():
        shutil.copy2(src, backup)

    out.to_csv(src, index=False)
    print(f"Wrote canonical dataset to {src} (backup at {backup}) with {len(out)} rows")


if __name__ == '__main__':
    main()
