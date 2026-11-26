"""
Porter Stemming Algorithm Subsystem for NovaSearch.
Normalizes English tokens to morphological root stems for enhanced recall and lexical matching.
"""

from typing import List


class PorterStemmer:
    """Implements Martin Porter's standard English stemming algorithm (1980)."""

    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        ch = word[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else not PorterStemmer._is_consonant(word, i - 1)
        return True

    @staticmethod
    def _measure(word: str) -> int:
        """Measures the number of consonant sequences between vowels [C](VC)^m[V]."""
        n = len(word)
        i = 0
        m = 0
        while i < n and PorterStemmer._is_consonant(word, i):
            i += 1
        while i < n:
            while i < n and not PorterStemmer._is_consonant(word, i):
                i += 1
            if i >= n:
                break
            while i < n and PorterStemmer._is_consonant(word, i):
                i += 1
            m += 1
        return m

    @staticmethod
    def _contains_vowel(word: str) -> bool:
        for i in range(len(word)):
            if not PorterStemmer._is_consonant(word, i):
                return True
        return False

    @staticmethod
    def _ends_double_consonant(word: str) -> bool:
        if len(word) < 2:
            return False
        return (
            word[-1] == word[-2]
            and PorterStemmer._is_consonant(word, len(word) - 1)
        )

    @staticmethod
    def _cvc(word: str) -> bool:
        """Checks if word ends in consonant-vowel-consonant, with last consonant not w, x, or y."""
        n = len(word)
        if n < 3:
            return False
        return (
            PorterStemmer._is_consonant(word, n - 1)
            and not PorterStemmer._is_consonant(word, n - 2)
            and PorterStemmer._is_consonant(word, n - 3)
            and word[-1] not in "wxy"
        )

    @classmethod
    def stem(cls, word: str) -> str:
        """Stems a single English word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif not word.endswith("ss") and word.endswith("s"):
            word = word[:-1]

        # Step 1b
        extra_step = False
        if word.endswith("eed"):
            stem = word[:-3]
            if cls._measure(stem) > 0:
                word = stem + "ee"
        elif word.endswith("ed"):
            stem = word[:-2]
            if cls._contains_vowel(stem):
                word = stem
                extra_step = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if cls._contains_vowel(stem):
                word = stem
                extra_step = True

        if extra_step:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word = word + "e"
            elif cls._ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif cls._measure(word) == 1 and cls._cvc(word):
                word = word + "e"

        # Step 1c
        if word.endswith("y") and cls._contains_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2: Suffix replacements when measure > 0
        step2_pairs = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, repl in step2_pairs:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if cls._measure(stem) > 0:
                    word = stem + repl
                break

        # Step 3
        step3_pairs = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, repl in step3_pairs:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if cls._measure(stem) > 0:
                    word = stem + repl
                break

        # Step 4
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if cls._measure(stem) > 1:
                    word = stem
                break
        else:
            if word.endswith("sion") or word.endswith("tion"):
                stem = word[:-3]
                if cls._measure(stem) > 1:
                    word = stem

        # Step 5a
        if word.endswith("e"):
            stem = word[:-1]
            m = cls._measure(stem)
            if m > 1 or (m == 1 and not cls._cvc(stem)):
                word = stem

        # Step 5b
        if cls._measure(word) > 1 and cls._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]

        return word

    @classmethod
    def transform(cls, tokens: List[str]) -> List[str]:
        """Stems a list of tokens."""
        return [cls.stem(t) for t in tokens]
