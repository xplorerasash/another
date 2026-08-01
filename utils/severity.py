"""Compute a severity score/label for a message by combining the trained
model's confidence with a lightweight keyword heuristic.

The label is primarily keyword-anchored so a mild insult is not escalated
to "severe" just because the classifier is very confident:
  - any severe keyword      -> "severe"
  - any moderate keyword    -> "moderate"
  - otherwise (no keyword)  -> confidence-based (severe/moderate/mild)

This keeps the blocking rules (3 violations in an hour, or 1 severe
violation) deterministic and sensible.
"""
import re
from typing import Dict

# Words/phrases that, on their own, strongly suggest a severe violation.
SEVERE_KEYWORDS = {
    "kill", "die", "suicide", "rape", "murder", "terrorist", "hate you",
}

# Words that suggest a milder, but still unwanted, insult.
MODERATE_KEYWORDS = {
    "ugly", "hate", "loser", "dumb", "shut up", "pathetic",
    "worthless", "useless", "stupid", "idiot",
}


def _has_keyword(text_lower: str, keyword: str) -> bool:
    """Match phrases as substrings and single words on word boundaries so
    "die" doesn't fire on "diet" or "died"."""
    if " " in keyword:
        return keyword in text_lower
    return re.search(rf"\b{re.escape(keyword)}\b", text_lower) is not None


def keyword_score(text: str) -> float:
    """Return a 0-1 heuristic score based on flagged keyword hits."""
    text_lower = text.lower()
    severe_hits = sum(1 for w in SEVERE_KEYWORDS if _has_keyword(text_lower, w))
    moderate_hits = sum(1 for w in MODERATE_KEYWORDS if _has_keyword(text_lower, w))
    return min(1.0, severe_hits * 0.5 + moderate_hits * 0.2)


def compute_severity(text: str, model_confidence: float, model_says_safe: bool = True) -> Dict:
    text_lower = text.lower()
    has_severe = any(_has_keyword(text_lower, w) for w in SEVERE_KEYWORDS)
    has_moderate = any(_has_keyword(text_lower, w) for w in MODERATE_KEYWORDS)
    kw_score = keyword_score(text)

    if model_says_safe and not has_severe and not has_moderate:
        return {
            "score": 0.0,
            "label": "safe",
            "model_confidence": round(model_confidence, 3),
            "keyword_score": 0.0,
        }

    if has_severe:
        label = "severe"
        score = max(0.9, model_confidence)
    elif has_moderate:
        label = "moderate"
        score = max(0.6, model_confidence)
    elif not model_says_safe:
        if model_confidence >= 0.8:
            label = "severe"
            score = model_confidence
        elif model_confidence >= 0.5:
            label = "moderate"
            score = model_confidence
        else:
            label = "mild"
            score = 0.5
    else:
        label = "safe"
        score = 0.0

    return {
        "score": round(min(score, 1.0), 3),
        "label": label,
        "model_confidence": round(model_confidence, 3),
        "keyword_score": round(kw_score, 3),
    }
