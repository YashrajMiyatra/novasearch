"""
Unit tests for NovaSearch MetadataFilter predicates.
"""

import unittest
from novasearch.filter import MetadataFilter


class TestMetadataFilter(unittest.TestCase):
    def test_empty_filter_matches_anything(self):
        doc_meta = {"category": "ml", "rating": 4.5}
        self.assertTrue(MetadataFilter.matches(doc_meta, {}))

    def test_exact_equality(self):
        doc_meta = {"author": "Yashraj", "year": 2026}
        self.assertTrue(MetadataFilter.matches(doc_meta, {"author": "Yashraj"}))
        self.assertFalse(MetadataFilter.matches(doc_meta, {"author": "Unknown"}))
        self.assertFalse(MetadataFilter.matches(doc_meta, {"missing": "field"}))

    def test_operators(self):
        doc_meta = {"rating": 4.8, "tags": ["search", "vector"], "year": 2026}
        self.assertTrue(MetadataFilter.matches(doc_meta, {"rating": {"$gte": 4.5}}))
        self.assertFalse(MetadataFilter.matches(doc_meta, {"rating": {"$lt": 4.0}}))
        self.assertTrue(MetadataFilter.matches(doc_meta, {"year": {"$ne": 2020}}))
        self.assertTrue(MetadataFilter.matches(doc_meta, {"rating": {"$gt": 4.0, "$lte": 5.0}}))


if __name__ == "__main__":
    unittest.main()
