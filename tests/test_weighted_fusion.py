"""
Unit tests for Weighted Reciprocal Rank Fusion and Ranking Evaluation Metrics.
"""

import unittest
from novasearch.fusion import reciprocal_rank_fusion
from novasearch.metrics import compute_mrr, compute_ndcg_at_k


class TestWeightedFusionAndMetrics(unittest.TestCase):
    def setUp(self):
        self.bm25_res = [
            {"id": "doc1", "text": "Python systems"},
            {"id": "doc2", "text": "Redis store"},
            {"id": "doc3", "text": "Autograd tensor"},
        ]
        self.vector_res = [
            {"id": "doc3", "text": "Autograd tensor"},
            {"id": "doc2", "text": "Redis store"},
            {"id": "doc4", "text": "Neural network"},
        ]

    def test_weighted_rrf_bias(self):
        # Heavily weight vector over BM25
        results_vector_heavy = reciprocal_rank_fusion(
            self.bm25_res, self.vector_res, k=10, weight_bm25=0.1, weight_vector=2.0
        )
        # doc3 was rank 1 in vector, so it should easily win
        self.assertEqual(results_vector_heavy[0]["id"], "doc3")

        # Heavily weight BM25 over vector
        results_bm25_heavy = reciprocal_rank_fusion(
            self.bm25_res, self.vector_res, k=10, weight_bm25=2.0, weight_vector=0.1
        )
        self.assertEqual(results_bm25_heavy[0]["id"], "doc1")

    def test_compute_mrr(self):
        rankings = [
            ["docA", "docB", "docC"],  # hit at rank 1 -> 1.0
            ["docX", "docY", "docB"],  # hit at rank 3 -> 0.3333
            ["docM", "docN", "docO"],  # no hit -> 0.0
        ]
        ground_truth = [
            {"docA"},
            {"docB"},
            {"docZ"},
        ]
        mrr = compute_mrr(rankings, ground_truth)
        expected = (1.0 + 1.0 / 3.0 + 0.0) / 3.0
        self.assertAlmostEqual(mrr, expected, places=4)

    def test_compute_ndcg(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}
        ndcg = compute_ndcg_at_k(retrieved, relevant, k=4)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)


if __name__ == "__main__":
    unittest.main()
