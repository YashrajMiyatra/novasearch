import unittest
from novasearch import LevenshteinAutomaton


class TestLevenshteinAutomaton(unittest.TestCase):
    def test_exact_and_single_edits(self):
        auto = LevenshteinAutomaton("banana", max_edits=1)
        self.assertTrue(auto.matches("banana"))    # 0 edits
        self.assertTrue(auto.matches("banan"))     # deletion
        self.assertTrue(auto.matches("bananas"))   # insertion
        self.assertTrue(auto.matches("banama"))    # substitution
        self.assertTrue(auto.matches("bandana"))   # insertion of 'd'

        self.assertFalse(auto.matches("apple"))
        self.assertFalse(auto.matches("bana"))     # 2 deletions
        self.assertFalse(auto.matches("bananamas")) # 2 insertions

    def test_two_edits(self):
        auto = LevenshteinAutomaton("search", max_edits=2)
        self.assertTrue(auto.matches("search"))
        self.assertTrue(auto.matches("serch"))     # 1 deletion
        self.assertTrue(auto.matches("srch"))      # 2 deletions
        self.assertTrue(auto.matches("research"))  # 2 insertions ('re')
        self.assertTrue(auto.matches("searched"))  # 2 insertions ('ed')
        self.assertFalse(auto.matches("unsearched")) # 4 insertions

    def test_filter_candidates(self):
        auto = LevenshteinAutomaton("kitten", max_edits=2)
        candidates = ["kitten", "sitting", "kitchen", "kitting", "dog", "puppy"]
        # "kitten" -> 0 edits
        # "kitchen" -> 2 edits (insert c, insert h)
        # "kitting" -> 2 edits (e->i, insert g)
        # "sitting" -> 3 edits (k->s, e->i, insert g)
        matches = auto.filter_candidates(candidates)
        self.assertIn("kitten", matches)
        self.assertIn("kitchen", matches)
        self.assertIn("kitting", matches)
        self.assertNotIn("sitting", matches)
        self.assertNotIn("dog", matches)
        self.assertNotIn("puppy", matches)

    def test_empty_string_matching(self):
        auto = LevenshteinAutomaton("", max_edits=2)
        self.assertTrue(auto.matches(""))
        self.assertTrue(auto.matches("a"))
        self.assertTrue(auto.matches("ab"))
        self.assertFalse(auto.matches("abc"))

    def test_invalid_max_edits(self):
        with self.assertRaises(ValueError):
            LevenshteinAutomaton("test", max_edits=-1)


if __name__ == "__main__":
    unittest.main()
