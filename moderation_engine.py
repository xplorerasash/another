"""Safety & Moderation Layer.

Given a raw message, this module runs it through the classifier, scores
its severity, and (if flagged) generates a respectful alternative. This is
the single entry point the rest of the app should use for "is this text
okay to send?" decisions -- both for user input and for the chatbot's own
generated replies (see response_filter.py).
"""
from typing import Dict

from moderation_model import get_model
from utils.severity import compute_severity
from utils.suggestion import suggest_alternative

# extras: language detection, keyword filtering, sentiment
try:
  from langdetect import detect
except Exception:
  def detect(text: str) -> str:  # type: ignore
    return "en"

try:
  from transformers import pipeline
  _sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  )
except Exception:
  _sentiment_pipeline = None

from utils.keywords import keyword_hits

BLOCKING_LABELS = {"moderate", "severe"}


def analyze_message(text: str) -> Dict:
    """Run the full moderation pipeline on a single message.

    Returns a dict with:
      text, is_harmful, model_confidence, severity_score, severity_label,
      suggested_alternative (str or None)
    """
    model = get_model()
    prediction = model.predict(text)

    # language detection
    lang = "en"
    try:
      lang = detect(text)
    except Exception:
      lang = "en"

    # quick keyword boost for non-english/bangla support
    kw_hits = keyword_hits(text, lang=lang)

    # optional sentiment check (if available): negative sentiment may increase severity
    sentiment_score = 0.0
    if _sentiment_pipeline is not None:
      try:
        s = _sentiment_pipeline(text[:512])[0]
        # convert labels like POSITIVE/NEGATIVE or LABEL_0 etc to a simple score
        if s.get("label") and s.get("score"):
          if s["label"].lower().startswith("neg"):
            sentiment_score = float(s["score"])  # 0..1
      except Exception:
        sentiment_score = 0.0

    is_harmful = prediction["is_harmful"]

    severity_info = compute_severity(text, prediction["confidence"], model_says_safe=not is_harmful)

    result = {
      "text": text,
      "language": lang,
      "is_harmful": is_harmful,
      "model_confidence": prediction["confidence"],
      "severity_score": severity_info["score"],
      "severity_label": severity_info["label"],
      "keyword_hits": kw_hits,
      "sentiment_score": round(sentiment_score, 3),
      "suggested_alternative": suggest_alternative(text) if is_harmful else None,
    }
    return result
