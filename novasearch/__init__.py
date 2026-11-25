"""
NovaSearch: Hybrid Search Engine combining BM25 Lexical Matching, Vector Search, and Reciprocal Rank Fusion.
"""

__version__ = "1.0.0"

from .engine import NovaEngine
from .bm25 import BM25Index, BM25PlusIndex
from .vector_index import VectorIndex
from .fusion import reciprocal_rank_fusion
from .crawler import FileCrawler
from .highlighter import highlight_snippets
from .tokenizer import tokenize, NgramTokenizer, StopwordFilter
from .metrics import compute_mrr, compute_ndcg_at_k
from .chunker import RecursiveTextSplitter, TextChunk
from .query_expansion import QueryExpander
from .filter import MetadataFilter
from .fuzzy import FuzzyMatcher, levenshtein_distance
from .jaccard import jaccard_similarity, token_jaccard, shingle_jaccard
from .vectorizer import CharNgramVectorizer
from .trie import TrieSuggester
from .spellcheck import SpellChecker
from .rewriter import QueryRewriter

__all__ = [
    "NovaEngine",
    "BM25Index",
    "BM25PlusIndex",
    "VectorIndex",
    "reciprocal_rank_fusion",
    "FileCrawler",
    "highlight_snippets",
    "tokenize",
    "NgramTokenizer",
    "StopwordFilter",
    "compute_mrr",
    "compute_ndcg_at_k",
    "RecursiveTextSplitter",
    "TextChunk",
    "QueryExpander",
    "MetadataFilter",
    "FuzzyMatcher",
    "levenshtein_distance",
    "jaccard_similarity",
    "token_jaccard",
    "shingle_jaccard",
    "CharNgramVectorizer",
    "TrieSuggester",
    "SpellChecker",
    "QueryRewriter",
]
