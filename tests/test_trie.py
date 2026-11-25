import unittest
from novasearch.trie import TrieSuggester


class TestTrieSuggester(unittest.TestCase):
    def setUp(self):
        self.trie = TrieSuggester()
        self.trie.insert("python", weight=10.0)
        self.trie.insert("pytorch", weight=8.0)
        self.trie.insert("pytest", weight=5.0)
        self.trie.insert("pyspark", weight=6.0)
        self.trie.insert("javascript", weight=9.0)

    def test_contains(self):
        self.assertTrue(self.trie.contains("python"))
        self.assertTrue(self.trie.contains("PyTorch"))
        self.assertFalse(self.trie.contains("pyth"))
        self.assertFalse(self.trie.contains("golang"))

    def test_prefix_suggestions_ranking(self):
        # Prefix "py" should return all 4 py* terms ranked by weight
        suggestions = self.trie.suggest("py", top_k=3)
        self.assertEqual(len(suggestions), 3)
        words = [w for w, _ in suggestions]
        self.assertEqual(words, ["python", "pytorch", "pyspark"])

    def test_empty_or_missing_prefix(self):
        self.assertEqual(self.trie.suggest(""), [])
        self.assertEqual(self.trie.suggest("xyz"), [])

    def test_size(self):
        self.assertEqual(self.trie.size, 5)


if __name__ == "__main__":
    unittest.main()
