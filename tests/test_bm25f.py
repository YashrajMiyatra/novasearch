import unittest
from novasearch import BM25FIndex


class TestBM25F(unittest.TestCase):
    def setUp(self):
        self.bm25f = BM25FIndex(
            k1=1.5,
            field_weights={"title": 3.0, "body": 1.0, "tags": 2.0},
            field_b={"title": 0.2, "body": 0.75, "tags": 0.1},
        )
        self.bm25f.add_document(
            doc_id="doc1",
            fields={
                "title": "FastAPI Web Framework",
                "body": "FastAPI is a modern fast web framework for building APIs with Python.",
                "tags": "python web api",
            },
        )
        self.bm25f.add_document(
            doc_id="doc2",
            fields={
                "title": "Flask Microframework",
                "body": "Flask is a lightweight WSGI web application framework in Python.",
                "tags": "python wsgi microframework",
            },
        )
        self.bm25f.add_document(
            doc_id="doc3",
            fields={
                "title": "Django Batteries Included",
                "body": "Django is a high-level Python web framework that encourages rapid development.",
                "tags": "python fullstack orm",
            },
        )

    def test_search_title_weight_boost(self):
        # Query matching "fastapi" in title vs generic body term
        results = self.bm25f.search("FastAPI", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertGreater(results[0]["score"], 0.0)

    def test_search_ranking_order(self):
        results = self.bm25f.search("Python web framework", top_k=3)
        self.assertEqual(len(results), 3)
        # All documents contain python and web, but ranking reflects field weights
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query(self):
        self.assertEqual(self.bm25f.search(""), [])

    def test_unmatched_term(self):
        self.assertEqual(self.bm25f.search("nonexistenttermxyz"), [])


if __name__ == "__main__":
    unittest.main()
