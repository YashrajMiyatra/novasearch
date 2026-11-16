"""
Local File & Documentation Crawler for NovaSearch.
Discovers and extracts text from Markdown, Python, Text, and JSON files.
"""

import os
from typing import Callable, Dict, List, Optional


class FileCrawler:
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".json", ".html", ".rst"}

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)

    def crawl(
        self,
        on_document: Optional[Callable[[str, str, Dict[str, str]], None]] = None,
    ) -> List[Dict[str, str]]:
        docs = []
        for root, _, files in os.walk(self.root_dir):
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue  # skip hidden dirs like .git
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        rel_path = os.path.relpath(fpath, self.root_dir)
                        doc_id = rel_path
                        meta = {
                            "path": fpath,
                            "filename": fname,
                            "extension": ext,
                            "size": str(len(content)),
                        }
                        if on_document:
                            on_document(doc_id, content, meta)
                        docs.append({"id": doc_id, "text": content, "metadata": meta})
                    except Exception:
                        pass
        return docs
