"""Suggest a safer way to phrase a flagged message."""
import random
import re

GENERIC_SUGGESTIONS = [
    "Try expressing what's bothering you without insults \u2014 what's the actual issue?",
    "That phrasing could hurt someone. Want to rephrase it more constructively?",
    "Let's keep this respectful. Could you say that in a calmer way?",
    "Consider focusing on the behavior you disagree with, not the person.",
]

# Naive word/phrase substitutions used to build a gentler rewrite when possible.
REPLACEMENTS = {
    "hate": "strongly dislike",
    "kill": "stop",
    "ugly": "not to my taste",
    "shut up": "please pause for a moment",
    "dumb": "confusing",
    "pathetic": "disappointing",
    "worthless": "not helpful right now",
    "useless": "not working for me",
}

DIRECT_RESPONSE_KEYWORDS = {
    "stupid": "I disagree with your response.",
    "idiot": "I disagree with you, but I respect your opinion.",
}


def suggest_alternative(text: str) -> str:
    """Return a rewritten, gentler version of the text when we recognize
    flagged words, otherwise fall back to a generic coaching message.
    """
    lowered = text.lower()

    for kw, response in DIRECT_RESPONSE_KEYWORDS.items():
        if kw in lowered:
            return response

    rewritten = text
    applied = False

    for bad, good in REPLACEMENTS.items():
        if bad in lowered:
            rewritten = re.sub(re.escape(bad), good, rewritten, flags=re.IGNORECASE)
            applied = True

    if applied:
        return f'Here\'s a gentler way to say that: "{rewritten}"'
    return random.choice(GENERIC_SUGGESTIONS)
