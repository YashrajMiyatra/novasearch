import unittest
from novasearch.bm25 import BM25Index


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index()
        self.index.add_document("doc1", "The quick brown fox jumps over the lazy dog")
        self.index.add_document("doc2", "Artificial intelligence and neural networks are transforming technology")
        self.index.add_document("doc3", "Brown bears love honey and jumping in the river")

    def test_search_relevance(self):
        results = self.index.search("brown fox", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc1")  # doc1 has both brown and fox

    def test_empty_query(self):
        self.assertEqual(self.index.search(""), [])


if __name__ == "__main__":
    unittest.main()
