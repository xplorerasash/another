"""Quality gate for the moderation model.

Minimum acceptable test-set metrics after retraining. The gate rejects a
retrain if any metric falls below its threshold, so a degraded model can
never replace the deployed one silently.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# Minimum acceptable held-out test metrics. Chosen with margin below the
# current fine-tuned model's scores (acc 0.9733, prec 0.9863, rec 0.96,
# f1 0.973, roc_auc 0.9966) so they only trip on meaningful degradation.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "accuracy": 0.92,
    "precision": 0.92,
    "recall": 0.90,
    "f1": 0.91,
    "roc_auc": 0.95,
}


def check_quality(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, Tuple[Optional[float], float]]]:
    """Compare a metrics dict against minimum thresholds.

    Returns ``(passed, failures)`` where ``failures`` maps each failing
    metric name to ``(actual, minimum)``. A metric that is missing is
    treated as a failure.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    failures: Dict[str, Tuple[Optional[float], float]] = {}
    for name, minimum in thresholds.items():
        actual = metrics.get(name)
        if actual is None:
            failures[name] = (None, float(minimum))
        elif float(actual) < float(minimum):
            failures[name] = (float(actual), float(minimum))
    return not failures, failures
