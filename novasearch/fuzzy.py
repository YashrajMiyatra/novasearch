"""
Fuzzy Query Matching and Typo Tolerance for NovaSearch.
Implements Wagner-Fischer Levenshtein distance dynamic programming matrix
and vocabulary candidate ranking.
"""

from typing import Dict, List, Optional, Set, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings using Wagner-Fischer DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]


class FuzzyMatcher:
    """
    Candidate suggestion engine that finds vocabulary matches within max_distance.
    """

    def __init__(self, vocabulary: Optional[List[str]] = None) -> None:
        self.vocabulary: Set[str] = set(vocabulary) if vocabulary else set()

    def add_term(self, term: str) -> None:
        self.vocabulary.add(term.lower())

    def add_terms(self, terms: List[str]) -> None:
        for t in terms:
            self.add_term(t)

    def suggest(self, term: str, max_distance: int = 2, top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Returns list of (candidate, distance) tuples sorted by ascending distance.
        """
        if not term:
            return []

        term_lower = term.lower()
        if term_lower in self.vocabulary:
            return [(term_lower, 0)]

        candidates = []
        for word in self.vocabulary:
            if abs(len(word) - len(term_lower)) > max_distance:
                continue
            dist = levenshtein_distance(term_lower, word)
            if dist <= max_distance:
                candidates.append((word, dist))

        candidates.sort(key=lambda x: (x[1], x[0]))
        return candidates[:top_k]
