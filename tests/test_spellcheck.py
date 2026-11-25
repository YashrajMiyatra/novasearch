import unittest
from novasearch.spellcheck import SpellChecker


class TestSpellChecker(unittest.TestCase):
    def setUp(self):
        self.checker = SpellChecker()
        corpus = [
            "python is a great programming language",
            "pytorch is a machine learning framework for python",
            "search engines index documents for retrieval",
            "python developers love clean code",
        ]
        self.checker.fit(corpus)

    def test_correct_spelling_unchanged(self):
        self.assertEqual(self.checker.correction("python"), "python")
        self.assertEqual(self.checker.correction("machine"), "machine")

    def test_single_edit_typo(self):
        # pythn -> python (deletion error)
        self.assertEqual(self.checker.correction("pythn"), "python")
        # pytoch -> pytorch (deletion error)
        self.assertEqual(self.checker.correction("pytoch"), "pytorch")
        # seach -> search (transposition / deletion)
        self.assertEqual(self.checker.correction("seach"), "search")

    def test_candidate_probability(self):
        self.assertGreater(self.checker.probability("python"), self.checker.probability("code"))


if __name__ == "__main__":
    unittest.main()
