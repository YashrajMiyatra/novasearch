"""
Statistical Spell Checker and Query Auto-Correction Engine for NovaSearch.
Implements Peter Norvig's probabilistic edit-distance model for noisy query correction.
"""

from collections import Counter
import re
from typing import Dict, List, Set, Tuple


class SpellChecker:
    """Probabilistic spelling correction engine based on word frequency and edit distance."""

    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self) -> None:
        self.words_count: Counter[str] = Counter()
        self.total_words: int = 0

    def fit(self, texts: List[str]) -> "SpellChecker":
        """Builds word frequency distribution from a list of documents."""
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            self.words_count.update(words)
        self.total_words = sum(self.words_count.values())
        return self

    def probability(self, word: str) -> float:
        """Probability of `word` in corpus."""
        return self.words_count[word] / max(self.total_words, 1)

    def correction(self, word: str) -> str:
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.probability)

    def candidates(self, word: str) -> Set[str]:
        """Generate possible spelling corrections for word."""
        word = word.lower()
        return (
            self._known({word})
            or self._known(self._edits1(word))
            or self._known(self._edits2(word))
            or {word}
        )

    def _known(self, words: Set[str]) -> Set[str]:
        """The subset of `words` that appear in the dictionary."""
        return {w for w in words if w in self.words_count}

    def _edits1(self, word: str) -> Set[str]:
        """All edits that are one edit away from `word`."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.LETTERS]
        inserts = [L + c + R for L, R in splits for c in self.LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> Set[str]:
        """All edits that are two edits away from `word`."""
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}
