"""
Unit tests for NovaSearch QueryExpander.
"""

import unittest
from novasearch.query_expansion import QueryExpander


class TestQueryExpander(unittest.TestCase):
    def setUp(self):
        self.expander = QueryExpander()

    def test_empty_query(self):
        self.assertEqual(self.expander.expand(""), "")

    def test_expansion_with_synonyms(self):
        query = "fast database"
        expanded = self.expander.expand(query, max_synonyms_per_term=2)
        # Should contain original terms plus synonyms
        self.assertIn("fast", expanded)
        self.assertIn("database", expanded)
        self.assertIn("quick", expanded)
        self.assertIn("datastore", expanded)

    def test_no_synonym_match(self):
        query = "zebra kangaroo"
        expanded = self.expander.expand(query)
        self.assertEqual(expanded, "zebra kangaroo")

    def test_custom_synonym_dictionary(self):
        custom = {"ml": ["machine-learning", "ai"]}
        expander = QueryExpander(synonym_map=custom)
        expanded = expander.expand("ml models")
        self.assertIn("machine-learning", expanded)
        self.assertIn("ai", expanded)


if __name__ == "__main__":
    unittest.main()
