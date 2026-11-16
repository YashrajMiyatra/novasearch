"""
Dense Vector Index for Semantic Search.
Computes Cosine Similarity between document and query embeddings.
"""

import math
from typing import Any, Dict, List


class VectorIndex:
    def __init__(self) -> None:
        self.vectors: Dict[str, List[float]] = {}
        self.norms: Dict[str, float] = {}

    def add_vector(self, doc_id: str, vector: List[float]) -> None:
        norm = math.sqrt(sum(x * x for x in vector))
        self.vectors[doc_id] = vector
        self.norms[doc_id] = norm

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0 or not self.vectors:
            return []

        scores = []
        for doc_id, vec in self.vectors.items():
            d_norm = self.norms[doc_id]
            if d_norm == 0:
                similarity = 0.0
            else:
                dot = sum(q * d for q, d in zip(query_vector, vec))
                similarity = dot / (q_norm * d_norm)
            scores.append((doc_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (doc_id, sim) in enumerate(scores[:top_k], 1):
            results.append({
                "id": doc_id,
                "score": round(sim, 4),
                "rank": rank,
            })
        return results
