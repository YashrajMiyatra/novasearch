"""
Reciprocal Rank Fusion (RRF) Engine.
Merges rankings from BM25 lexical search and Dense Vector semantic search.
Formula: RRF_score(d) = sum(1 / (k + rank(d)))
"""

from collections import defaultdict
from typing import Any, Dict, List


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combines lexical and vector rankings using Reciprocal Rank Fusion.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_details: Dict[str, Dict[str, Any]] = {}

    # 1. Process BM25 rankings
    for rank, item in enumerate(bm25_results, 1):
        doc_id = item["id"]
        rrf_scores[doc_id] += 1.0 / (k + rank)
        doc_details[doc_id] = item

    # 2. Process Vector rankings
    for rank, item in enumerate(vector_results, 1):
        doc_id = item["id"]
        rrf_scores[doc_id] += 1.0 / (k + rank)
        if doc_id not in doc_details:
            doc_details[doc_id] = item

    # Sort descending by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for rank, (doc_id, score) in enumerate(sorted_items, 1):
        info = doc_details[doc_id]
        results.append({
            "id": doc_id,
            "rrf_score": round(score, 6),
            "rank": rank,
            "text": info.get("text", ""),
            "metadata": info.get("metadata", {}),
        })

    return results
