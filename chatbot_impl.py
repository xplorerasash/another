"""Coordinates the complete SafeChat-AI message-processing pipeline.

    User Message
         |
    Is the user currently blocked? --yes--> return block notice
         | no
    Safety & Moderation Layer (moderation_engine.analyze_message)
         |
    Safe? --no--> record violation, return warning + suggested alternative
         | yes
    Conversation Layer (conversation_engine.generate_response)
         |
    Response Filter (response_filter.filter_response)
         |
    Return final reply to the user
"""
import logging
from typing import Dict, List, Optional

from moderation_engine import analyze_message
from violation_manager import ViolationManager
from conversation_engine import generate_response
from response_filter import filter_response

logger = logging.getLogger("safechat.chatbot")

_tracker = ViolationManager()


def process_message(
    user_id: str,
    message: str,
    history: Optional[List[Dict]] = None,
    tracker: Optional[ViolationManager] = None,
) -> Dict:
    """Run one message through the full pipeline and return a structured
    result for the UI layer (app.py) to render.

    Result shape:
      {"type": "blocked_user" | "blocked_message" | "chat",
       "reply": str,
       "analysis": dict | None,
       "violation_info": dict | None}
    """
    tracker = tracker or _tracker

    if tracker.is_blocked(user_id):
        remaining = tracker.seconds_until_unblocked(user_id)
        return {
            "type": "blocked_user",
            "reply": (
                f"\U0001F6AB You are temporarily blocked for {remaining} more "
                "seconds due to repeated violations."
            ),
            "analysis": None,
            "violation_info": None,
        }

    analysis = analyze_message(message)

    if analysis["is_harmful"]:
        violation_info = tracker.register_violation(user_id, analysis["severity_label"])

        severity = analysis["severity_label"].capitalize()
        confidence = analysis["model_confidence"]

        reply = (
            f"\u26A0\uFE0F Your message was blocked because it contains offensive language. "
            f"Severity: {severity}. "
            f"Please use respectful language."
        )
        if analysis["suggested_alternative"]:
            reply += f" Suggested alternative: {analysis['suggested_alternative']}"

        if violation_info["blocked"]:
            reply += (
                f"\n\n\U0001F6AB You have been temporarily blocked due to repeated violations."
            )

        return {
            "type": "blocked_message",
            "reply": reply,
            "analysis": {
                **analysis,
                "violation_count": violation_info["violation_count"],
            },
            "violation_info": violation_info,
        }

    raw_reply = generate_response(message, history)
    safe_reply = filter_response(raw_reply)

    return {
        "type": "chat",
        "reply": safe_reply,
        "analysis": analysis,
        "violation_info": None,
    }