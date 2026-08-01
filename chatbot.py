"""SafeChat-AI main chatbot class.

Re-exports `chatbot_impl.process_message` as the module-level
`process_message` function and provides the `SafeChatbot` class
for use by both the terminal interface (chatbot.py.new) and tests.
"""
from typing import Dict, List, Optional

from chatbot_impl import process_message

__all__ = ["SafeChatbot", "process_message"]


class SafeChatbot:
    """High-level chatbot wrapper with optional model-path injection.

    Every message, including greetings, runs through the full two-layer
    pipeline: Safety & Moderation first, then Conversation (if safe).
    """

    def __init__(self, model_path=None):
        if model_path is not None:
            try:
                import moderation_model as _mm
                _mm._default_model = _mm.ModerationModel(model_path)
            except Exception:
                pass

    def process_message(
        self,
        message: str,
        user_id: str = "test_user",
        history: Optional[List[Dict]] = None,
        tracker=None,
    ) -> Dict:
        return process_message(user_id, message, history, tracker=tracker)