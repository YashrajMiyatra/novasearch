"""
Information Retrieval (IR) Evaluation Metrics.
Calculates Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG@K).
"""

import math
from typing import List, Set, Union


def compute_mrr(rankings: List[List[Union[str, int]]], ground_truth: List[Set[Union[str, int]]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across multiple queries.
    MRR = (1 / |Q|) * sum(1 / rank_i)
    """
    if not rankings or not ground_truth or len(rankings) != len(ground_truth):
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(rankings, ground_truth):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def compute_ndcg_at_k(retrieved: List[Union[str, int]], relevant: Set[Union[str, int]], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K).
    """
    top_k = retrieved[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1)

    # Ideal DCG (IDCG)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum((2.0 ** 1.0 - 1.0) / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0
