"""Multilingual keyword lists and helper utilities for quick filtering.
"""
from typing import Set

# English keywords (expanded)
EN_KEYWORDS: Set[str] = {
    "kill",
    "die",
    "suicide",
    "rape",
    "murder",
    "terrorist",
    "stupid",
    "idiot",
    "ugly",
    "hate",
    "loser",
    "dumb",
    "shut up",
    "pathetic",
    "worthless",
    "useless",
}

# Bangla keywords (UTF-8, common abusive words). This is illustrative;
# for production you should expand and validate with native speakers.
BN_KEYWORDS: Set[str] = {
    "মরা",  # die
    "কোথা",  # nonsense placeholder
    "বোকা",  # idiot
    "গাধা",  # donkey (insult)
    "ভালো না",  # not good
}


def keyword_hits(text: str, lang: str = "en") -> int:
    low = text.lower()
    if lang and lang.startswith("bn"):
        return sum(1 for k in BN_KEYWORDS if k in low)
    return sum(1 for k in EN_KEYWORDS if k in low)
