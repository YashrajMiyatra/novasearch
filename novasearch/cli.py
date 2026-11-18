"""
NovaSearch Interactive CLI.
Command-line search interface with live query execution and snippet highlighting.
"""

import argparse
import os
import sys
from .engine import NovaEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaSearch: Hybrid Search Engine CLI")
    parser.add_argument("-v", "--version", action="version", version="NovaSearch 1.0.0")
    parser.add_argument("path", nargs="?", default=".", help="Directory of documents to index and search")
    parser.add_argument("-q", "--query", help="Single query to run non-interactively")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("-m", "--mode", default="bm25", choices=["bm25", "hybrid"], help="Search mode")

    args = parser.parse_args()
    target_dir = os.path.abspath(args.path)

    if not os.path.exists(target_dir):
        print(f"Error: Path '{target_dir}' does not exist.")
        sys.exit(1)

    print(f"\033[1;36m=== NovaSearch Hybrid Engine (v1.0.0) ===\033[0m")
    print(f"[*] Crawling and indexing documents from: \033[1;34m{target_dir}\033[0m...")

    engine = NovaEngine()
    count = engine.crawl_and_index(target_dir)
    print(f"[+] Indexed \033[1;32m{count}\033[0m documents across corpus.\n")

    if args.query:
        run_query(engine, args.query, args.top_k, args.mode)
        return

    # Interactive mode
    print("Type your search query below. Type 'exit' or press Ctrl+C to quit.\n")
    while True:
        try:
            query = input("\033[1;32mnovasearch>\033[0m ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            run_query(engine, query, args.top_k, args.mode)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


def run_query(engine: NovaEngine, query: str, top_k: int, mode: str) -> None:
    results = engine.search(query, mode=mode, top_k=top_k)
    if not results:
        print("  \033[90m(No matching documents found)\033[0m\n")
        return

    print(f"\nFound \033[1;32m{len(results)}\033[0m matching document(s):")
    for r in results:
        score_val = r.get("rrf_score") or r.get("score")
        print(f"\033[1;36m#{r['rank']}\033[0m \033[1m{r['id']}\033[0m  \033[90m[score: {score_val}]\033[0m")
        print(f"   {r['snippet']}")
        print()


if __name__ == "__main__":
    main()
