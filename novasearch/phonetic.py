"""
Phonetic Analysis Subsystem for NovaSearch.
Implements the American Soundex algorithm for pronunciation-based token normalization and fuzzy phonetic retrieval.
"""

from typing import List, Optional


class SoundexFilter:
    """Encodes terms into American Soundex codes for phonetic search matching."""

    _MAP = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    @classmethod
    def encode(cls, word: str) -> str:
        """
        Computes standard 4-character American Soundex code (e.g. 'Robert' -> 'R163').
        """
        if not word:
            return ""

        clean = "".join(c.upper() for c in word if c.isalpha())
        if not clean:
            return ""

        first_char = clean[0]
        first_code = cls._MAP.get(first_char, "")

        codes = [first_char]
        prev_code = first_code

        for char in clean[1:]:
            code = cls._MAP.get(char, "")
            if code != "":
                if code != prev_code:
                    codes.append(code)
                prev_code = code
            elif char in ("H", "W"):
                # H and W do not act as separators for duplicate digits
                pass
            else:
                # Vowels (A, E, I, O, U, Y) act as separators
                prev_code = ""

        res = codes[0] + "".join(codes[1:])
        return res[:4].ljust(4, "0")

    @classmethod
    def transform(cls, tokens: List[str]) -> List[str]:
        """Transforms a stream of tokens into their corresponding Soundex phonetic codes."""
        return [cls.encode(t) for t in tokens if cls.encode(t)]

    @classmethod
    def is_phonetic_match(cls, word1: str, word2: str) -> bool:
        """Checks if two words share the same phonetic Soundex representation."""
        c1 = cls.encode(word1)
        c2 = cls.encode(word2)
        return bool(c1 and c2 and c1 == c2)
