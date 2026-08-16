"""
Text preprocessing utilities.

Mirrors the exact preprocessing pipeline used in
notebooks/01_NLP_Preprocessing.ipynb so that runtime text is
transformed the same way the training data was:
    lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize
"""
from __future__ import annotations

import string
import threading

import nltk

_NLTK_READY = False
_NLTK_LOCK = threading.Lock()


def ensure_nltk_data() -> None:
    """Download required NLTK corpora once (idempotent, thread-safe)."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    with _NLTK_LOCK:
        if _NLTK_READY:
            return
        for pkg, path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("stopwords", "corpora/stopwords"),
            ("wordnet", "corpora/wordnet"),
            ("omw-1.4", "corpora/omw-1.4"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        _NLTK_READY = True


def preprocess_text(text: str) -> str:
    """Reproduce the exact cleaning pipeline used to build `processed_text`
    columns in every dataset and to train the ML/DL intent classifiers."""
    ensure_nltk_data()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
