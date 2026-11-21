"""Benchmark lexical BM25 vs Reciprocal Rank Fusion latency."""
import time
from novasearch.bm25 import BM25Index
from novasearch.fusion import reciprocal_rank_fusion


def benchmark_search(n_docs: int = 5000):
    docs = [f"Document number {i} discussing systems programming and search engine design" for i in range(n_docs)]
    index = BM25Index()
    index.index_documents(docs)

    print(f"Benchmarking retrieval over {n_docs} documents...")
    start = time.perf_counter()
    results = index.search("systems programming", top_k=10)
    bm25_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    fused = reciprocal_rank_fusion([results, results], top_n=10)
    fusion_time = (time.perf_counter() - start) * 1000

    print(f"BM25 Search: {bm25_time:.2f} ms | RRF Fusion: {fusion_time:.2f} ms")


if __name__ == "__main__":
    benchmark_search()
