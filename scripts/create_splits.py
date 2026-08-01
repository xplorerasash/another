"""Create 70/15/15 train/validation/test splits from dataset/labeled.csv.

This is a lightweight helper if you want to create splits without running
the full HF training (which downloads models).
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    src = Path('dataset') / 'labeled.csv'
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    df = pd.read_csv(src).dropna(subset=['text', 'label'])
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    processed_dir = Path('processed')
    processed_dir.mkdir(exist_ok=True)
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    val_df.to_csv(processed_dir / 'validation.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    print(f"Created splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


if __name__ == '__main__':
    main()
