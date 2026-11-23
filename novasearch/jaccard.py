"""
Jaccard Similarity and Shingle Overlap Metric for NovaSearch.
Computes token-level and character-level Jaccard set intersection over union.
"""

from typing import List, Set, Union


def jaccard_similarity(set_a: Union[Set, List], set_b: Union[Set, List]) -> float:
    """
    Computes Jaccard similarity coefficient J(A, B) = |A ∩ B| / |A ∪ B|.
    Returns 1.0 for two empty sets, and a float in [0.0, 1.0] otherwise.
    """
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 1.0
    union_len = len(a | b)
    if union_len == 0:
        return 1.0
    return len(a & b) / union_len


def token_jaccard(text_a: str, text_b: str) -> float:
    """Computes Jaccard similarity between whitespace-tokenized texts."""
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    return jaccard_similarity(tokens_a, tokens_b)


def shingle_jaccard(text_a: str, text_b: str, k: int = 3) -> float:
    """
    Computes character k-shingle Jaccard overlap between two strings.
    """
    def _get_shingles(s: str, shingle_k: int) -> Set[str]:
        s_clean = "".join(s.lower().split())
        if len(s_clean) < shingle_k:
            return {s_clean} if s_clean else set()
        return {s_clean[i : i + shingle_k] for i in range(len(s_clean) - shingle_k + 1)}

    return jaccard_similarity(_get_shingles(text_a, k), _get_shingles(text_b, k))
