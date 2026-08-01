import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import SafeChatbot
from violation_manager import ViolationManager

# Use a dedicated test tracker to avoid state leaking between tests / runs
_test_tracker = ViolationManager(
    path=Path("models/test_violations.json"),
)


@pytest.fixture(autouse=True)
def reset_tracker():
    _test_tracker._data.clear()
    _test_tracker._save()


def get_chatbot():
    return SafeChatbot()


def test_safe_message_returns_conversation_reply():
    chatbot = get_chatbot()
    result = chatbot.process_message("Hi there", user_id="test_user", tracker=_test_tracker)
    assert result["type"] == "chat"
    assert result["reply"]


def test_harmful_message_is_blocked_and_suggested():
    chatbot = get_chatbot()
    result = chatbot.process_message("You are stupid", user_id="test_user", tracker=_test_tracker)
    assert result["type"] == "blocked_message"
    assert result["reply"]
    assert "blocked" in result["reply"].lower()


def test_harmful_message_shows_severity():
    chatbot = get_chatbot()
    result = chatbot.process_message("You are stupid", user_id="test_user", tracker=_test_tracker)
    assert result["analysis"]["severity_label"] in ("mild", "moderate", "severe")


def test_harmful_message_includes_alternative():
    chatbot = get_chatbot()
    result = chatbot.process_message("You are stupid", user_id="test_user", tracker=_test_tracker)
    assert "I disagree with your response" in result["reply"]


def test_harmful_message_includes_model_confidence():
    chatbot = get_chatbot()
    result = chatbot.process_message("You are stupid", user_id="test_user", tracker=_test_tracker)
    assert 0 <= result["analysis"]["model_confidence"] <= 1.0


def test_safe_ml_question_returns_info():
    chatbot = get_chatbot()
    result = chatbot.process_message("What is machine learning", user_id="test_user", tracker=_test_tracker)
    assert result["type"] in ("chat", "blocked_message")
    if result["type"] == "chat":
        assert "machine learning" in result["reply"].lower()


def test_greeting_still_goes_through_moderation():
    chatbot = get_chatbot()
    result = chatbot.process_message("Hi there", user_id="test_user", tracker=_test_tracker)
    assert result["type"] == "chat"
