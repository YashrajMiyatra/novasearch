import unittest
from novasearch.rewriter import QueryRewriter


class TestQueryRewriter(unittest.TestCase):
    def setUp(self):
        self.rewriter = QueryRewriter()

    def test_default_acronym_expansion(self):
        q = "deploy k8s cluster"
        self.assertEqual(self.rewriter.rewrite(q), "deploy kubernetes cluster")

        q2 = "nlp and ml algorithms"
        self.assertEqual(
            self.rewriter.rewrite(q2),
            "natural language processing and machine learning algorithms",
        )

    def test_custom_rule(self):
        self.rewriter.add_rule(r"\bpostgres\b", "postgresql")
        self.assertEqual(
            self.rewriter.rewrite("connect to postgres"),
            "connect to postgresql",
        )

    def test_empty_string(self):
        self.assertEqual(self.rewriter.rewrite(""), "")

    def test_batch_rewrite(self):
        queries = ["k8s pods", "ml models"]
        res = self.rewriter.batch_rewrite(queries)
        self.assertEqual(res, ["kubernetes pods", "machine learning models"])


if __name__ == "__main__":
    unittest.main()
