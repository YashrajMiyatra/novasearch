"""
Unit tests for NovaSearch RecursiveTextSplitter and chunking pipeline.
"""

import unittest
from novasearch.chunker import RecursiveTextSplitter, TextChunk


class TestRecursiveTextSplitter(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            RecursiveTextSplitter(max_chunk_size=50, chunk_overlap=50)

    def test_empty_string(self):
        splitter = RecursiveTextSplitter(max_chunk_size=100, chunk_overlap=20)
        self.assertEqual(splitter.split_text(""), [])

    def test_small_text_single_chunk(self):
        splitter = RecursiveTextSplitter(max_chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text("Hello world! This is a simple test.", metadata={"doc_id": "doc_1"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello world! This is a simple test.")
        self.assertEqual(chunks[0].metadata["doc_id"], "doc_1")
        self.assertEqual(chunks[0].index, 0)

    def test_paragraph_and_sentence_splitting(self):
        text = (
            "Paragraph one is moderately long and discusses hybrid search engines.\n\n"
            "Paragraph two goes into reciprocal rank fusion and vector embeddings.\n\n"
            "Paragraph three summarizes the retrieval evaluation metrics like MRR and NDCG."
        )
        splitter = RecursiveTextSplitter(max_chunk_size=90, chunk_overlap=15)
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 120)  # bounded chunk sizes

    def test_chunk_indices_and_monotonicity(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        splitter = RecursiveTextSplitter(max_chunk_size=30, chunk_overlap=10)
        chunks = splitter.split_text(text)
        for i, c in enumerate(chunks):
            self.assertEqual(c.index, i)
            self.assertGreater(len(c.text), 0)


if __name__ == "__main__":
    unittest.main()
