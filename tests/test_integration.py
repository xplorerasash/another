import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import SafeChatbot
from violation_manager import ViolationManager

_test_tracker = ViolationManager(
    path=Path("models/test_violations.json"),
)


@pytest.fixture(autouse=True)
def reset_tracker():
    _test_tracker._data.clear()
    _test_tracker._save()


def get_chatbot():
    return SafeChatbot()


def test_normal_conversation_flows_to_chat_with_no_violation():
    result = get_chatbot().process_message(
        "Hi there, how are you?", user_id="conv_user", tracker=_test_tracker
    )
    assert result["type"] == "chat"
    assert result["reply"]
    assert result["violation_info"] is None
    assert result["analysis"]["is_harmful"] is False


def test_mild_insult_is_blocked_as_moderate():
    result = get_chatbot().process_message(
        "You are stupid", user_id="insult_user", tracker=_test_tracker
    )
    assert result["type"] == "blocked_message"
    assert result["analysis"]["severity_label"] == "moderate"
    assert "blocked" in result["reply"].lower()


def test_bullying_is_blocked_and_includes_alternative():
    result = get_chatbot().process_message(
        "You are worthless and pathetic", user_id="bully_user", tracker=_test_tracker
    )
    assert result["type"] == "blocked_message"
    assert "suggested alternative" in result["reply"].lower()


def test_threat_is_blocked():
    result = get_chatbot().process_message(
        "You better watch your back", user_id="threat_user", tracker=_test_tracker
    )
    assert result["type"] == "blocked_message"
    assert result["reply"]


def test_violation_count_increments_across_messages():
    bot = get_chatbot()
    first = bot.process_message("You are stupid", user_id="count_user", tracker=_test_tracker)
    second = bot.process_message("You are stupid", user_id="count_user", tracker=_test_tracker)
    assert first["analysis"]["violation_count"] == 1
    assert second["analysis"]["violation_count"] == 2
    assert second["violation_info"]["blocked"] is False


def test_three_violations_triggers_temporary_block():
    bot = get_chatbot()
    third = bot.process_message("You are stupid", user_id="block_user", tracker=_test_tracker)
    bot.process_message("You are stupid", user_id="block_user", tracker=_test_tracker)
    third = bot.process_message("You are stupid", user_id="block_user", tracker=_test_tracker)
    assert third["analysis"]["violation_count"] == 3
    assert third["violation_info"]["blocked"] is True
    assert "temporarily blocked" in third["reply"].lower()


def test_blocked_user_receives_blocked_notice_even_for_safe_message():
    bot = get_chatbot()
    for _ in range(3):
        bot.process_message("You are stupid", user_id="later_user", tracker=_test_tracker)
    result = bot.process_message("Hi there", user_id="later_user", tracker=_test_tracker)
    assert result["type"] == "blocked_user"
    assert "blocked" in result["reply"].lower()


def test_single_severe_violation_blocks_immediately():
    result = get_chatbot().process_message(
        "I hate you", user_id="severe_user", tracker=_test_tracker
    )
    assert result["type"] == "blocked_message"
    assert result["analysis"]["severity_label"] == "severe"
    assert result["violation_info"]["blocked"] is True


def test_all_blocked_messages_expose_confidence():
    bot = get_chatbot()
    for msg, uid in [
        ("You are stupid", "conf_user"),
        ("You are worthless and pathetic", "conf_user"),
        ("I hate you", "conf_user"),
    ]:
        result = bot.process_message(msg, user_id=uid, tracker=_test_tracker)
        assert result["type"] == "blocked_message"
        assert 0 <= result["analysis"]["model_confidence"] <= 1.0
