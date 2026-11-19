import unittest
from novasearch.fusion import reciprocal_rank_fusion


class TestRRF(unittest.TestCase):
    def test_fusion_ordering(self):
        # Doc A is rank 1 in lexical, rank 2 in vector
        # Doc B is rank 2 in lexical, rank 1 in vector
        # Doc C is rank 3 in lexical, absent in vector
        lexical = [{"id": "docA", "text": "a"}, {"id": "docB", "text": "b"}, {"id": "docC", "text": "c"}]
        vector = [{"id": "docB", "text": "b"}, {"id": "docA", "text": "a"}]

        fused = reciprocal_rank_fusion(lexical, vector, k=60)
        # Both docA and docB have (1/61 + 1/62), docC has 1/63
        self.assertEqual(len(fused), 3)
        self.assertIn(fused[0]["id"], ("docA", "docB"))
        self.assertEqual(fused[2]["id"], "docC")


    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([], []), [])

    def test_disjoint_rankings(self):
        lexical = [{"id": "doc1", "text": "lexical match"}]
        vector = [{"id": "doc2", "text": "semantic match"}]
        fused = reciprocal_rank_fusion(lexical, vector, k=60)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_top_n_truncation(self):
        lexical = [{"id": f"doc_{i}", "text": f"text {i}"} for i in range(15)]
        fused = reciprocal_rank_fusion(lexical, [], top_n=5)
        self.assertEqual(len(fused), 5)


if __name__ == "__main__":
    unittest.main()
