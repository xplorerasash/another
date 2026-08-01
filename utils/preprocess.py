"""Text cleaning and preprocessing utilities shared by training, evaluation,
the terminal chatbot, and the web app.
"""
import re

import emoji
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure the required NLTK corpora are available (downloads once, then reuses cache).
for resource, lookup_path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(lookup_path)
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, expand contractions, strip emojis/URLs/mentions/punctuation,
    remove stopwords, and lemmatize the remaining tokens.
    """
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    words = []
    for word in text.split():
        if word not in STOPWORDS:
            words.append(lemmatizer.lemmatize(word))
    return " ".join(words)


def preprocess_text(text: str) -> str:
    """Alias kept for backwards compatibility with earlier scripts."""
    return clean_text(text)
