import unittest
from novasearch.jaccard import jaccard_similarity, token_jaccard, shingle_jaccard


class TestJaccardSimilarity(unittest.TestCase):
    def test_jaccard_similarity_sets(self):
        a = {"apple", "banana", "cherry"}
        b = {"banana", "cherry", "date"}
        # Intersection = 2, Union = 4 -> 0.5
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.5)

    def test_token_jaccard(self):
        s1 = "quick brown fox"
        s2 = "brown fox jumps"
        # Shared: brown, fox (2). Total: 4 -> 0.5
        self.assertAlmostEqual(token_jaccard(s1, s2), 0.5)

    def test_shingle_jaccard(self):
        # Identical strings have shingle similarity 1.0
        self.assertAlmostEqual(shingle_jaccard("search", "search", k=3), 1.0)
        # Disjoint strings
        self.assertAlmostEqual(shingle_jaccard("abc", "xyz", k=3), 0.0)

    def test_empty_inputs(self):
        self.assertAlmostEqual(jaccard_similarity(set(), set()), 1.0)


if __name__ == "__main__":
    unittest.main()
