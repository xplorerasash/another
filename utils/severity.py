"""Compute a severity score/label for a message by combining the trained
model's confidence with a lightweight keyword heuristic.
"""
from typing import Dict

# Words that, on their own, strongly suggest a severe violation.
SEVERE_KEYWORDS = {
    "kill", "die", "suicide", "rape", "murder", "terrorist",
}

# Words that suggest a milder, but still unwanted, insult.
MODERATE_KEYWORDS = {
    "ugly", "hate", "loser", "dumb", "shut up", "pathetic",
    "worthless", "useless", "stupid", "idiot",
}


def keyword_score(text: str) -> float:
    """Return a 0-1 heuristic score based on flagged keyword hits."""
    text_lower = text.lower()
    severe_hits = sum(1 for w in SEVERE_KEYWORDS if w in text_lower)
    moderate_hits = sum(1 for w in MODERATE_KEYWORDS if w in text_lower)
    return min(1.0, severe_hits * 0.5 + moderate_hits * 0.2)


def compute_severity(text: str, model_confidence: float, model_says_safe: bool = True) -> Dict:
    kw_score = keyword_score(text)

    if model_says_safe and kw_score == 0:
        return {
            "score": 0.0,
            "label": "safe",
            "model_confidence": round(model_confidence, 3),
            "keyword_score": 0.0,
        }

    combined = round(min(max(model_confidence, kw_score), 1.0), 3)

    if combined >= 0.8:
        label = "severe"
    elif combined >= 0.5:
        label = "moderate"
    elif combined > 0.2:
        label = "mild"
    else:
        label = "safe"

    return {
        "score": combined,
        "label": label,
        "model_confidence": round(model_confidence, 3),
        "keyword_score": round(kw_score, 3),
    }
