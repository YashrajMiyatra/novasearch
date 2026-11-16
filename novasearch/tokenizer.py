"""
Text normalization, tokenization, stopword removal, and stemming.
Zero external dependencies required.
"""

import re
from typing import List, Set

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def simple_stem(word: str) -> str:
    """Lightweight rule-based English suffix stemmer."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 4:
        return word[:-3]
    elif word.endswith("ed") and len(word) > 3:
        return word[:-2]
    elif word.endswith("es") and len(word) > 3:
        return word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """Tokenizes text into cleaned and stemmed terms."""
    # Find all alphanumeric sequences
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    tokens = []
    for t in raw_tokens:
        if remove_stopwords and t in STOPWORDS:
            continue
        if stem:
            t = simple_stem(t)
        if t:
            tokens.append(t)
    return tokens
