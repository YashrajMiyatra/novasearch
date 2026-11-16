"""
NovaEngine: Master Hybrid Search Engine Orchestrator.
Combines BM25 Lexical Search, Dense Vector Semantic Search, and Reciprocal Rank Fusion.
"""

from typing import Any, Dict, List, Optional
from .bm25 import BM25Index
from .crawler import FileCrawler
from .fusion import reciprocal_rank_fusion
from .highlighter import highlight_snippets
from .vector_index import VectorIndex


class NovaEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.bm25 = BM25Index(k1=k1, b=b)
        self.vector_index = VectorIndex()

    def index_document(
        self,
        doc_id: str,
        text: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bm25.add_document(doc_id, text, metadata=metadata)
        if vector is not None:
            self.vector_index.add_vector(doc_id, vector)

    def crawl_and_index(self, directory_path: str) -> int:
        crawler = FileCrawler(directory_path)
        docs = crawler.crawl(on_document=self.bm25.add_document)
        return len(docs)

    def search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Executes search using 'hybrid', 'bm25', or 'vector' mode.
        """
        mode = mode.lower()

        if mode == "bm25" or (mode == "hybrid" and query_vector is None):
            results = self.bm25.search(query, top_k=top_k)
            for r in results:
                r["snippet"] = highlight_snippets(r["text"], query)
            return results

        elif mode == "vector":
            if query_vector is None:
                raise ValueError("query_vector is required for mode='vector'")
            results = self.vector_index.search(query_vector, top_k=top_k)
            for r in results:
                doc = self.bm25.docs.get(r["id"], {})
                r["text"] = doc.get("text", "")
                r["metadata"] = doc.get("metadata", {})
                r["snippet"] = highlight_snippets(r["text"], query)
            return results

        elif mode == "hybrid":
            bm25_res = self.bm25.search(query, top_k=top_k * 2)
            vector_res = self.vector_index.search(query_vector, top_k=top_k * 2)
            fused = reciprocal_rank_fusion(bm25_res, vector_res, top_n=top_k)
            for r in fused:
                r["snippet"] = highlight_snippets(r["text"], query)
            return fused

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'hybrid', 'bm25', or 'vector'.")
