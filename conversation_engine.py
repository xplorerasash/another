"""Conversation Layer.

Handles messages that the Safety & Moderation Layer has already confirmed
are safe. This starter implementation is a simple intent-matching engine,
but it's written behind a single function -- generate_response(text,
history) -> str -- so it can later be swapped for a local or hosted
generative language model (e.g. a fine-tuned GPT-2/DialoGPT, or a call to
an LLM API) without changing chatbot.py, response_filter.py, or app.py.
"""
import random
import re
from typing import Dict, List, Optional

# Each intent maps to a list of regex patterns and a list of candidate replies.
INTENTS = {
    "compliment": {
        "patterns": [r"\byou are (good|great|helpful|awesome|kind|nice|amazing)\b",
                     r"\byou('?re| are) the best\b",
                     r"\bi like you\b"],
        "responses": [
            "Thank you! That's very kind of you \U0001F60A How can I help you today?",
            "I appreciate that! \U0001F60A Let me know what you need.",
        ],
    },
    "how_are_you": {
        "patterns": [r"how are you", r"how('?s| is) it going"],
        "responses": [
            "Hello! I'm doing well, thank you for asking. How can I help you today?",
            "Hi! I'm doing great, thanks for asking. What can I do for you?",
        ],
    },
    "greeting": {
        "patterns": [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b"],
        "responses": [
            "Hello! \U0001F44B Welcome to SafeChat. How can I help you today?",
            "Hi there! What can I do for you?",
        ],
    },
    "identity": {
        "patterns": [r"who are you", r"your name", r"\bwhat are you\b"],
        "responses": [
            "I'm SafeChat-AI, a chatbot designed to promote safer online communication.",
        ],
    },
    "capabilities": {
        "patterns": [r"what can you do", r"help me with", r"your (purpose|job)"],
        "responses": [
            "I can chat with you about all kinds of things, and I also keep the "
            "conversation respectful by flagging harmful or offensive messages.",
        ],
    },
    "thanks": {
        "patterns": [r"\bthank(s| you)\b", r"\bthx\b"],
        "responses": ["You're welcome! \U0001F60A", "Anytime! Glad I could help."],
    },
    "farewell": {
        "patterns": [r"\bbye\b", r"goodbye", r"see (ya|you)"],
        "responses": ["Goodbye! Have a great day!", "See you later! Take care."],
    },
    "programming_question": {
        "patterns": [r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\bcoding\b", r"\bprogramming\b"],
        "responses": [
            "Python is a popular programming language used for web development, "
            "data science, artificial intelligence, automation, and more.",
            "Programming is the process of creating instructions that tell a computer "
            "how to perform specific tasks. Is there a particular language you're interested in?",
        ],
    },
    "ml_question": {
        "patterns": [r"machine learning"],
        "responses": [
            "Machine learning is a branch of artificial intelligence that allows "
            "computers to learn patterns from data and make predictions or decisions.",
        ],
    },
    "ai_question": {
        "patterns": [r"\bartificial intelligence\b", r"\bwhat is ai\b"],
        "responses": [
            "Artificial intelligence refers to computer systems designed to perform "
            "tasks that normally require human intelligence, such as understanding "
            "language, recognizing patterns, and making decisions.",
        ],
    },
    "bert_question": {
        "patterns": [r"\bbert\b"],
        "responses": [
            "BERT is a transformer-based language model that's widely used for text "
            "classification tasks \u2014 it's the same kind of model that powers my "
            "safety layer.",
        ],
    },
}

FALLBACK_RESPONSES = [
    "That's interesting! Could you tell me more?",
    "I see. What else would you like to talk about?",
    "Got it -- is there something specific I can help you with?",
]


def _match_intent(text: str) -> Optional[str]:
    lowered = text.lower()
    for intent, spec in INTENTS.items():
        for pattern in spec["patterns"]:
            if re.search(pattern, lowered):
                return intent
    return None


def generate_response(text: str, history: Optional[List[Dict]] = None) -> str:
    """Generate a reply to a message that has already been confirmed safe.

    `history` is a list of {"role": "user"/"assistant", "text": str} dicts
    representing prior turns in the conversation. The rule-based engine
    doesn't use it yet, but it's threaded through the whole pipeline so a
    future generative backend can use it for context without any changes
    to chatbot.py or app.py.
    """
    intent = _match_intent(text)
    if intent is not None:
        return random.choice(INTENTS[intent]["responses"])
    return random.choice(FALLBACK_RESPONSES)
