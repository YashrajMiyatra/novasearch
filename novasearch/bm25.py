"""
Okapi BM25 Inverted Index & Scoring Engine.
Implements modern Lucene/Elasticsearch-style BM25 ranking.

Robertson-Spärck Jones Scoring Model:
- Term Saturation (k1=1.5): Governs how rapidly term frequency dampens relevance gains.
- Document Normalization (b=0.75): Penalizes overly verbose documents relative to avg_doc_len.
- RSJ Inverse Document Frequency: log(1 + (N - n + 0.5) / (n + 0.5))
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from .tokenizer import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0

        # inverted_index: term -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        # doc_lens: doc_id -> total token count
        self.doc_lens: Dict[str, int] = {}
        # docs: doc_id -> metadata
        self.docs: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        tokens = tokenize(text)
        doc_len = len(tokens)
        self.doc_lens[doc_id] = doc_len
        self.docs[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
        }

        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            self.inverted_index[term][doc_id] = freq

        self.corpus_size = len(self.docs)
        self.avg_doc_len = sum(self.doc_lens.values()) / max(self.corpus_size, 1)

    def _idf(self, term: str) -> float:
        """Robertson-Spärck Jones IDF calculation with floor."""
        doc_freq = len(self.inverted_index.get(term, {}))
        if doc_freq == 0:
            return 0.0
        # Standard Lucene/BM25 IDF formula
        return math.log(1.0 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            postings = self.inverted_index[term]

            for doc_id, tf in postings.items():
                doc_len = self.doc_lens[doc_id]
                # Length normalization
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
                scores[doc_id] += term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_id, score) in enumerate(ranked, 1):
            doc = self.docs[doc_id]
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "rank": rank,
                "text": doc["text"],
                "metadata": doc["metadata"],
            })
        return results


class BM25PlusIndex(BM25Index):
    """
    BM25+ Ranking Algorithm (Lv & Zhai, 2011).
    Adds a lower-bound tuning parameter delta (default 1.0) to term saturation,
    preventing over-penalization of long documents containing relevant query terms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> None:
        super().__init__(k1=k1, b=b)
        self.delta = delta

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            postings = self.inverted_index[term]

            for doc_id, tf in postings.items():
                doc_len = self.doc_lens[doc_id]
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                term_score = idf * (((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)) + self.delta)
                scores[doc_id] += term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_id, score) in enumerate(ranked, 1):
            doc = self.docs[doc_id]
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "rank": rank,
                "text": doc["text"],
                "metadata": doc["metadata"],
            })
        return results


class BM25FIndex:
    """
    Multi-field BM25F ranking algorithm (Robertson et al., 2004).
    Calculates combined pseudo-term frequencies across heterogeneous fields
    (e.g., title, body, keywords) using field-specific length normalizations (b_f)
    and field boost weights (w_f).
    """

    def __init__(
        self,
        k1: float = 1.5,
        field_weights: Optional[Dict[str, float]] = None,
        field_b: Optional[Dict[str, float]] = None,
    ) -> None:
        self.k1 = k1
        self.field_weights = field_weights or {"title": 2.0, "body": 1.0}
        self.field_b = field_b or {"title": 0.3, "body": 0.75}

        self.corpus_size = 0
        # doc_field_lens[field][doc_id] = length
        self.doc_field_lens: Dict[str, Dict[str, int]] = defaultdict(dict)
        # avg_field_lens[field] = float
        self.avg_field_lens: Dict[str, float] = defaultdict(float)

        # inverted_index[term][doc_id][field] = freq
        self.inverted_index: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        # docs[doc_id] = {"fields": fields, "metadata": metadata}
        self.docs: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc_id: str, fields: Dict[str, str], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.docs[doc_id] = {
            "id": doc_id,
            "fields": fields,
            "metadata": metadata or {},
        }

        for field, text in fields.items():
            tokens = tokenize(text or "")
            flen = len(tokens)
            self.doc_field_lens[field][doc_id] = flen
            tf_counter = Counter(tokens)
            for term, freq in tf_counter.items():
                self.inverted_index[term][doc_id][field] = freq

        self.corpus_size = len(self.docs)
        self._recompute_avg_field_lens()

    def _recompute_avg_field_lens(self) -> None:
        for field, doc_lens in self.doc_field_lens.items():
            if self.corpus_size > 0:
                self.avg_field_lens[field] = sum(doc_lens.values()) / self.corpus_size
            else:
                self.avg_field_lens[field] = 0.0

    def _idf(self, term: str) -> float:
        n = len(self.inverted_index.get(term, {}))
        return math.log(1.0 + (self.corpus_size - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            doc_postings = self.inverted_index[term]

            for doc_id, field_tfs in doc_postings.items():
                # Compute pseudo-frequency across weighted fields
                pseudo_tf = 0.0
                for field, tf in field_tfs.items():
                    w_f = self.field_weights.get(field, 1.0)
                    b_f = self.field_b.get(field, 0.75)
                    avg_len = self.avg_field_lens.get(field, 1.0) or 1.0
                    doc_len = self.doc_field_lens[field].get(doc_id, 0)

                    len_norm = 1.0 - b_f + b_f * (doc_len / avg_len)
                    pseudo_tf += w_f * (tf / len_norm if len_norm > 0 else tf)

                term_score = idf * (pseudo_tf / (self.k1 + pseudo_tf))
                scores[doc_id] += term_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_id, score) in enumerate(ranked, 1):
            doc = self.docs[doc_id]
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "rank": rank,
                "fields": doc["fields"],
                "metadata": doc["metadata"],
            })
        return results
