"""
Search Snippet Extractor & Keyword Highlighter.
Produces context-aware excerpts with highlighted search terms.
"""

import re
from typing import List
from .tokenizer import tokenize


def highlight_snippets(
    text: str,
    query: str,
    max_length: int = 160,
    ansi: bool = True,
) -> str:
    """
    Extracts the most relevant snippet from text containing query terms,
    highlighting matching keywords.
    """
    query_terms = tokenize(query, remove_stopwords=True, stem=False)
    if not query_terms:
        return text[:max_length] + ("..." if len(text) > max_length else "")

    # Split into sentences or lines
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    best_sentence = ""
    max_matches = -1

    for s in sentences:
        s_lower = s.lower()
        matches = sum(1 for q in query_terms if q in s_lower)
        if matches > max_matches:
            max_matches = matches
            best_sentence = s.strip()

    if not best_sentence:
        best_sentence = text[:max_length]

    # Truncate if too long
    if len(best_sentence) > max_length:
        best_sentence = best_sentence[: max_length - 3] + "..."

    # Highlight terms
    highlighted = best_sentence
    for q in query_terms:
        pattern = re.compile(rf"\b({re.escape(q)}\w*)\b", re.IGNORECASE)
        if ansi:
            highlighted = pattern.sub(r"\033[1;33m\1\033[0m", highlighted)
        else:
            highlighted = pattern.sub(r"<b>\1</b>", highlighted)

    return highlighted
