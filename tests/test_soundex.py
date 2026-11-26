import unittest
from novasearch import SoundexFilter


class TestSoundexFilter(unittest.TestCase):
    def test_canonical_soundex_names(self):
        # Classic phonetic pairs
        self.assertEqual(SoundexFilter.encode("Robert"), "R163")
        self.assertEqual(SoundexFilter.encode("Rupert"), "R163")
        self.assertTrue(SoundexFilter.is_phonetic_match("Robert", "Rupert"))

        self.assertEqual(SoundexFilter.encode("Smith"), "S530")
        self.assertEqual(SoundexFilter.encode("Smythe"), "S530")
        self.assertTrue(SoundexFilter.is_phonetic_match("Smith", "Smythe"))

        self.assertEqual(SoundexFilter.encode("Euler"), "E460")
        self.assertEqual(SoundexFilter.encode("Ellery"), "E460")
        self.assertTrue(SoundexFilter.is_phonetic_match("Euler", "Ellery"))

        self.assertEqual(SoundexFilter.encode("Gauss"), "G200")
        self.assertEqual(SoundexFilter.encode("Ghosh"), "G200")
        self.assertTrue(SoundexFilter.is_phonetic_match("Gauss", "Ghosh"))

        self.assertEqual(SoundexFilter.encode("Hilbert"), "H416")
        self.assertEqual(SoundexFilter.encode("Heilbronn"), "H416")
        self.assertTrue(SoundexFilter.is_phonetic_match("Hilbert", "Heilbronn"))

    def test_transform_tokens(self):
        tokens = ["quick", "brown", "fox", "jumps"]
        codes = SoundexFilter.transform(tokens)
        self.assertEqual(len(codes), 4)
        for code in codes:
            self.assertEqual(len(code), 4)
            self.assertTrue(code[0].isalpha())
            self.assertTrue(code[1:].isdigit())

    def test_empty_and_special_characters(self):
        self.assertEqual(SoundexFilter.encode(""), "")
        self.assertEqual(SoundexFilter.encode("12345"), "")
        self.assertEqual(SoundexFilter.encode("!@#$%^"), "")
        # Mixed with non-letters
        self.assertEqual(SoundexFilter.encode("O'Connor"), SoundexFilter.encode("OConnor"))

    def test_padding_and_truncation(self):
        # Short word pads with zeros
        self.assertEqual(SoundexFilter.encode("A"), "A000")
        self.assertEqual(SoundexFilter.encode("Day"), "D000")
        # Long word truncates to 4 chars
        code = SoundexFilter.encode("Washington")
        self.assertEqual(len(code), 4)
        self.assertEqual(code, "W252")


if __name__ == "__main__":
    unittest.main()
