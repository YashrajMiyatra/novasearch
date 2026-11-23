"""
Text Chunking and Segmentation for RAG / Dense Indexing in NovaSearch.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional


class TextChunk:
    def __init__(self, text: str, index: int, start_idx: int, end_idx: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.text = text
        self.index = index
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"TextChunk(index={self.index}, len={len(self.text)}, range=[{self.start_idx}:{self.end_idx}])"


class RecursiveTextSplitter:
    """
    Splits text recursively by separators: paragraphs, sentences, words, and characters
    to guarantee chunks stay under max_chunk_size with overlap.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, max_chunk_size: int = 200, chunk_overlap: int = 40, separators: Optional[List[str]] = None) -> None:
        if chunk_overlap >= max_chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than max_chunk_size ({max_chunk_size})")
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Splits raw text into a sequence of TextChunk instances."""
        if not text:
            return []

        raw_splits = self._split_recursive(text, self.separators)
        # Merge splits up to max_chunk_size with overlap
        chunks: List[TextChunk] = []
        current_chunk: List[str] = []
        current_len = 0
        current_start = 0
        chunk_idx = 0

        for segment in raw_splits:
            if not segment:
                continue
            seg_len = len(segment)
            if current_len + seg_len > self.max_chunk_size and current_chunk:
                joined = "".join(current_chunk)
                end_pos = current_start + len(joined)
                chunks.append(TextChunk(joined, chunk_idx, current_start, end_pos, metadata))
                chunk_idx += 1

                # Keep overlap from the end
                overlap_chars = []
                overlap_len = 0
                for part in reversed(current_chunk):
                    if overlap_len + len(part) <= self.chunk_overlap:
                        overlap_chars.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break
                current_chunk = overlap_chars
                current_len = overlap_len
                current_start = end_pos - overlap_len

            current_chunk.append(segment)
            current_len += seg_len

        if current_chunk:
            joined = "".join(current_chunk)
            chunks.append(TextChunk(joined, chunk_idx, current_start, current_start + len(joined), metadata))

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.max_chunk_size or not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result: List[str] = []
        for i, s in enumerate(splits):
            chunk = s + (sep if i < len(splits) - 1 else "")
            if len(chunk) > self.max_chunk_size and remaining_seps:
                result.extend(self._split_recursive(chunk, remaining_seps))
            else:
                result.append(chunk)

        return result
