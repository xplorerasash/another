import json
from pathlib import Path

import pandas as pd

from moderation_model import get_model
from model_quality import DEFAULT_THRESHOLDS, check_quality


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


def test_quality_gate_accepts_current_metrics():
    metrics = {"accuracy": 0.9733, "precision": 0.9863, "recall": 0.96, "f1": 0.973, "roc_auc": 0.9966}
    passed, failures = check_quality(metrics)
    assert passed, f"Current metrics must pass the gate, got failures: {failures}"
    assert failures == {}


def test_quality_gate_rejects_degraded_metrics():
    metrics = {"accuracy": 0.80, "precision": 0.90, "recall": 0.75, "f1": 0.82, "roc_auc": 0.85}
    passed, failures = check_quality(metrics)
    assert not passed
    assert set(failures) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert failures["accuracy"][0] == 0.80
    assert failures["accuracy"][1] == 0.92


def test_quality_gate_rejects_missing_metric():
    passed, failures = check_quality({"accuracy": 0.99})
    assert not passed
    assert "f1" in failures
    assert failures["f1"][0] is None


def test_current_eval_report_passes_quality_gate():
    report_path = Path("models/eval_report.json")
    assert report_path.exists(), "eval_report.json should exist from scripts/retrain.py"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    metrics = {name: report[name] for name in DEFAULT_THRESHOLDS}
    passed, failures = check_quality(metrics)
    assert passed, (
        "Deployed model must meet minimum quality thresholds. "
        f"Failures: {failures}. Retrain with `python scripts/train_and_evaluate.py`."
    )
