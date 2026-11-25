"""
Character N-Gram Vectorizer for Subword Lexical-Dense Matching in NovaSearch.
Decomposes tokens into boundary-tagged character shingles (e.g. '<search>')
and computes L2-normalized sparse subword representations for typo-resilient retrieval.
"""

import math
from collections import Counter
from typing import Dict, List, Set, Tuple


class CharNgramVectorizer:
    """Extracts boundary-padded character n-grams and computes normalized feature vectors."""

    def __init__(self, min_n: int = 3, max_n: int = 3) -> None:
        self.min_n = min_n
        self.max_n = max_n
        self.vocabulary: Dict[str, int] = {}
        self.doc_count = 0
        self.doc_freqs: Dict[str, int] = Counter()

    def extract_ngrams(self, text: str) -> List[str]:
        """Extracts boundary-marked n-grams from input text."""
        tokens = text.lower().split()
        ngrams: List[str] = []
        for token in tokens:
            padded = f"<{token}>"
            for n in range(self.min_n, self.max_n + 1):
                if len(padded) < n:
                    ngrams.append(padded)
                else:
                    for i in range(len(padded) - n + 1):
                        ngrams.append(padded[i : i + n])
        return ngrams

    def fit(self, documents: List[str]) -> "CharNgramVectorizer":
        """Fits vocabulary and document frequencies over corpus."""
        self.doc_count = len(documents)
        self.doc_freqs.clear()
        vocab_set: Set[str] = set()

        for doc in documents:
            ngrams = set(self.extract_ngrams(doc))
            for ng in ngrams:
                self.doc_freqs[ng] += 1
                vocab_set.add(ng)

        self.vocabulary = {ng: idx for idx, ng in enumerate(sorted(vocab_set))}
        return self

    def transform(self, text: str, use_idf: bool = True) -> Dict[str, float]:
        """Converts text into an L2-normalized sparse vector of n-grams."""
        ngrams = self.extract_ngrams(text)
        if not ngrams:
            return {}

        counts = Counter(ngrams)
        raw_vec: Dict[str, float] = {}

        for ng, tf in counts.items():
            if self.vocabulary and ng not in self.vocabulary:
                continue
            if use_idf and self.doc_count > 0:
                df = self.doc_freqs.get(ng, 1)
                idf = math.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0
                raw_vec[ng] = float(tf) * idf
            else:
                raw_vec[ng] = float(tf)

        norm = math.sqrt(sum(v * v for v in raw_vec.values()))
        if norm == 0.0:
            return {}
        return {k: round(v / norm, 6) for k, v in raw_vec.items()}

    @staticmethod
    def similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Computes cosine similarity between two sparse normalized vectors."""
        if not vec1 or not vec2:
            return 0.0
        # Iterate over smaller dictionary for O(min(|vec1|, |vec2|)) speed
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
        return sum(v * vec2.get(k, 0.0) for k, v in vec1.items())
