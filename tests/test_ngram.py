import unittest
from novasearch.tokenizer import NgramTokenizer


class TestNgramTokenizer(unittest.TestCase):
    def test_char_ngrams(self):
        tokenizer = NgramTokenizer(min_n=3, max_n=4, ngram_type="char")
        ngrams = tokenizer.tokenize("search")
        # 'search':
        # 3-grams: 'sea', 'ear', 'arc', 'rch'
        # 4-grams: 'sear', 'earc', 'arch'
        self.assertIn("sea", ngrams)
        self.assertIn("rch", ngrams)
        self.assertIn("sear", ngrams)
        self.assertEqual(len(ngrams), 7)

    def test_word_ngrams(self):
        tokenizer = NgramTokenizer(min_n=2, max_n=2, ngram_type="word")
        ngrams = tokenizer.tokenize("quick brown fox jumps")
        self.assertEqual(ngrams, ["quick brown", "brown fox", "fox jumps"])

    def test_empty_string(self):
        tokenizer = NgramTokenizer()
        self.assertEqual(tokenizer.tokenize(""), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            NgramTokenizer(min_n=5, max_n=3)
        with self.assertRaises(ValueError):
            NgramTokenizer(ngram_type="invalid")


if __name__ == "__main__":
    unittest.main()
