import unittest
from novasearch.vectorizer import CharNgramVectorizer


class TestCharNgramVectorizer(unittest.TestCase):
    def test_extract_ngrams(self):
        vec = CharNgramVectorizer(min_n=3, max_n=3)
        ngrams = vec.extract_ngrams("cat")
        # padded: <cat> -> <ca, cat, at>
        self.assertEqual(ngrams, ["<ca", "cat", "at>"])

    def test_fit_and_transform(self):
        docs = [
            "distributed systems consensus",
            "distributed storage database",
            "microservices architecture network",
        ]
        vectorizer = CharNgramVectorizer(min_n=3, max_n=3)
        vectorizer.fit(docs)
        self.assertGreater(len(vectorizer.vocabulary), 10)

        v1 = vectorizer.transform("distributed systems")
        v2 = vectorizer.transform("distributed architecture")
        v3 = vectorizer.transform("totally unrelated query")

        sim1_2 = vectorizer.similarity(v1, v2)
        sim1_3 = vectorizer.similarity(v1, v3)

        self.assertGreater(sim1_2, 0.2)
        self.assertLess(sim1_3, sim1_2)

    def test_typo_resilience(self):
        vectorizer = CharNgramVectorizer(min_n=3, max_n=3)
        vectorizer.fit(["elasticsearch search engine"])

        v_exact = vectorizer.transform("elasticsearch")
        v_typo = vectorizer.transform("elastiksearch")  # 1 typo 'k' instead of 'c'

        sim = vectorizer.similarity(v_exact, v_typo)
        self.assertGreater(sim, 0.6)  # High subword overlap


if __name__ == "__main__":
    unittest.main()
