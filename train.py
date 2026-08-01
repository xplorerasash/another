"""Train the SafeChat-AI cyberbullying detection model.

Reads a labeled dataset from `dataset/`, cleans the text, splits it into
train/test sets (saved under `processed/`), fits a TF-IDF + Logistic
Regression pipeline, and saves the trained model to `models/`.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = [
    BASE_DIR / "dataset" / "labeled.csv",
    BASE_DIR / "dataset" / "labeled_data.csv",
    BASE_DIR / "dataset" / "cyberbullying_tweets.csv",
]
PROCESSED_DIR = BASE_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "cyberbullying_model.joblib"

SAFE_CLASSES = {"not_cyberbullying", "normal", "neutral", "none", "safe", "neither", "non-bullying", "0", 0, "2", 2}


def resolve_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No supported dataset file was found in the dataset/ folder. "
        f"Expected one of: {[str(p) for p in DATASET_CANDIDATES]}"
    )


def map_label(value) -> int:
    """Map arbitrary label formats (numeric class ids, strings, etc.) to a
    binary target: 0 = safe, 1 = bullying/offensive.
    """
    value_str = str(value).strip().lower()

    if value_str in SAFE_CLASSES:
        return 0

    if value_str in {"hate", "hate_speech", "offensive", "offensive_language", "bullying", "abusive", "1", "yes", "true"}:
        return 1

    try:
        numeric_value = int(float(value))
    except (TypeError, ValueError):
        numeric_value = None

    if numeric_value is not None:
        # Kaggle "labeled_data.csv" convention: 0=hate, 1=offensive, 2=neither
        return 0 if numeric_value == 2 else 1

    return 0 if value_str in {"safe", "neither", "normal", "neutral", "none", "non-bullying"} else 1


def load_dataset() -> pd.DataFrame:
    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)

    text_col = next(
        (col for col in df.columns if col.lower() in {"tweet", "text", "content", "message", "comment"}),
        None,
    )
    label_col = next(
        (col for col in df.columns if col.lower() in {"class", "label", "type", "target", "category"}),
        None,
    )

    if not text_col or not label_col:
        raise ValueError(
            "The dataset must include a text column (tweet/text/content/message/comment) "
            "and a label column (class/label/type/target/category)."
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].apply(map_label)
    df["clean_text"] = df["text"].apply(clean_text)

    df = df.dropna(subset=["text", "label"]).drop_duplicates().reset_index(drop=True)
    df = df[df["clean_text"].str.len() > 5]
    return df.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved processed data to {TRAIN_PATH} and {TEST_PATH}")
    return train_df, test_df


def train_and_save_model():
    df = load_dataset()
    print(f"Loaded {len(df)} labeled rows.")
    print(df["label"].value_counts(normalize=True).rename("proportion"))

    train_df, test_df = save_processed_data(df)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(train_df["text"], train_df["label"])

    predictions = model.predict(test_df["text"])
    print(classification_report(test_df["label"], predictions, target_names=["safe", "bullying"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_model()
