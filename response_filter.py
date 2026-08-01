"""Filters chatbot-generated responses through the moderation engine
before they are shown to the user. This guarantees SafeChat-AI never
displays a harmful reply, even if the conversation engine (rule-based
today, possibly generative tomorrow) produces something inappropriate.
"""
from moderation_engine import analyze_message

SAFE_FALLBACK_RESPONSE = (
    "Sorry, I couldn't come up with an appropriate response to that. "
    "Could you rephrase your message?"
)


def filter_response(candidate_response: str) -> str:
    """Return candidate_response unchanged if it passes moderation,
    otherwise return a safe fallback message."""
    analysis = analyze_message(candidate_response)
    if analysis["is_harmful"]:
        return SAFE_FALLBACK_RESPONSE
    return candidate_response
