import unittest
from novasearch.bm25 import BM25Index, BM25PlusIndex


class TestBM25Plus(unittest.TestCase):
    def setUp(self):
        self.standard = BM25Index()
        self.plus = BM25PlusIndex(delta=1.0)

        # Short doc with query term
        self.short_doc = "python microservices architecture"
        # Very long doc with query term and lots of unrelated text
        self.long_doc = "python " + " unrelated words padding test sentence " * 50

        self.standard.add_document("doc_short", self.short_doc)
        self.standard.add_document("doc_long", self.long_doc)

        self.plus.add_document("doc_short", self.short_doc)
        self.plus.add_document("doc_long", self.long_doc)

    def test_bm25_plus_score_floor(self):
        # In BM25+, the long document score receives a floor boost of delta * idf
        std_results = {r["id"]: r["score"] for r in self.standard.search("python")}
        plus_results = {r["id"]: r["score"] for r in self.plus.search("python")}

        # BM25+ scores must be higher than standard BM25 for matched documents due to delta
        self.assertGreater(plus_results["doc_long"], std_results["doc_long"])
        self.assertGreater(plus_results["doc_short"], std_results["doc_short"])

    def test_empty_query(self):
        self.assertEqual(self.plus.search(""), [])

    def test_non_existent_term(self):
        self.assertEqual(self.plus.search("nonexistentterm123"), [])


if __name__ == "__main__":
    unittest.main()
