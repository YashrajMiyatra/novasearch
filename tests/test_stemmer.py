import unittest
from novasearch import PorterStemmer


class TestPorterStemmer(unittest.TestCase):
    def test_step1a_plurals(self):
        self.assertEqual(PorterStemmer.stem("cats"), "cat")
        self.assertEqual(PorterStemmer.stem("ponies"), "poni")
        self.assertEqual(PorterStemmer.stem("caresses"), "caress")
        self.assertEqual(PorterStemmer.stem("cats"), "cat")

    def test_step1b_past_and_participle(self):
        self.assertEqual(PorterStemmer.stem("running"), "run")
        self.assertEqual(PorterStemmer.stem("troubled"), "troubl")
        self.assertEqual(PorterStemmer.stem("agreed"), "agre")
        self.assertEqual(PorterStemmer.stem("meetings"), "meet")

    def test_step1c_y_to_i(self):
        self.assertEqual(PorterStemmer.stem("happy"), "happi")
        self.assertEqual(PorterStemmer.stem("sky"), "sky")

    def test_derivations_and_suffixes(self):
        self.assertEqual(PorterStemmer.stem("generalization"), "gener")
        self.assertEqual(PorterStemmer.stem("conditional"), "condit")
        self.assertEqual(PorterStemmer.stem("sensibility"), "sensibl")
        self.assertEqual(PorterStemmer.stem("hopeful"), "hope")

    def test_transform_tokens(self):
        tokens = ["The", "connections", "are", "running", "smoothly"]
        stemmed = PorterStemmer.transform(tokens)
        self.assertEqual(stemmed, ["the", "connect", "ar", "run", "smoothli"])

    def test_short_words(self):
        self.assertEqual(PorterStemmer.stem("a"), "a")
        self.assertEqual(PorterStemmer.stem("to"), "to")
        self.assertEqual(PorterStemmer.stem(""), "")


if __name__ == "__main__":
    unittest.main()
