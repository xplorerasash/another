from pathlib import Path

import pandas as pd

from moderation_model import get_model


def test_balanced_dataset_is_present_and_balanced():
    dataset_path = Path("dataset/balanced.csv")
    assert dataset_path.exists(), "Balanced dataset should be generated before training"

    df = pd.read_csv(dataset_path)
    counts = df["label"].value_counts().sort_index().to_dict()
    assert counts.get(0, 0) > 0
    assert counts.get(1, 0) > 0
    assert abs(counts[0] - counts[1]) <= max(counts[0], counts[1]) * 0.05


def test_moderation_model_classifies_safe_and_harmful_messages():
    model = get_model()
    safe_result = model.predict("Hello there, how are you doing today?")
    harmful_result = model.predict("You are stupid and worthless")

    assert safe_result["is_harmful"] is False
    assert harmful_result["is_harmful"] is True
    assert 0.0 <= safe_result["confidence"] <= 1.0
    assert 0.0 <= harmful_result["confidence"] <= 1.0
