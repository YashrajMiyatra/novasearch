import unittest
from novasearch.fuzzy import FuzzyMatcher, levenshtein_distance


class TestFuzzyMatching(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("search", "search"), 0)
        self.assertEqual(levenshtein_distance("book", "back"), 2)
        self.assertEqual(levenshtein_distance("", "hello"), 5)

    def test_fuzzy_matcher_exact_match(self):
        matcher = FuzzyMatcher(["python", "javascript", "golang"])
        results = matcher.suggest("python")
        self.assertEqual(results, [("python", 0)])

    def test_fuzzy_matcher_suggestions(self):
        matcher = FuzzyMatcher(["pytorch", "python", "pyramid", "torch"])
        # 'pytrch' has distance 1 to 'pytorch'
        suggestions = matcher.suggest("pytrch", max_distance=1)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0], ("pytorch", 1))

    def test_fuzzy_matcher_empty(self):
        matcher = FuzzyMatcher()
        self.assertEqual(matcher.suggest(""), [])


if __name__ == "__main__":
    unittest.main()
