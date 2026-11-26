import unittest
from novasearch.synonyms import SynonymFilter


class TestSynonymFilter(unittest.TestCase):
    def setUp(self):
        self.filter = SynonymFilter({
            "automobile": ["car", "vehicle"],
            "fast": ["quick", "rapid"],
        })

    def test_expand_tokens(self):
        tokens = ["fast", "car"]
        expanded = self.filter.expand(tokens)
        self.assertIn("fast", expanded)
        self.assertIn("quick", expanded)
        self.assertIn("car", expanded)
        self.assertIn("automobile", expanded)

    def test_canonicalize(self):
        # Among {"automobile", "car", "vehicle"}, lexicographically first is "automobile"
        self.assertEqual(self.filter.canonicalize("car"), "automobile")
        self.assertEqual(self.filter.canonicalize("vehicle"), "automobile")
        # Unmapped word returns itself
        self.assertEqual(self.filter.canonicalize("python"), "python")


if __name__ == "__main__":
    unittest.main()
