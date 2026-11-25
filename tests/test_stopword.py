import unittest
from novasearch.tokenizer import StopwordFilter


class TestStopwordFilter(unittest.TestCase):
    def test_default_stopword_filtering(self):
        f = StopwordFilter()
        tokens = ["the", "quick", "brown", "fox", "and", "a", "dog"]
        filtered = f.filter(tokens)
        self.assertEqual(filtered, ["quick", "brown", "fox", "dog"])

    def test_custom_stopwords(self):
        f = StopwordFilter(custom_stopwords={"custom", "junk"}, include_defaults=False)
        tokens = ["the", "custom", "apple", "junk", "banana"]
        filtered = f.filter(tokens)
        self.assertEqual(filtered, ["the", "apple", "banana"])

    def test_add_and_remove_stopwords(self):
        f = StopwordFilter(include_defaults=False)
        f.add_stopwords(["foo", "bar"])
        self.assertEqual(f.filter(["foo", "baz", "bar"]), ["baz"])

        f.remove_stopwords(["foo"])
        self.assertEqual(f.filter(["foo", "baz", "bar"]), ["foo", "baz"])


if __name__ == "__main__":
    unittest.main()
