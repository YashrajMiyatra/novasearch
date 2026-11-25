"""
Prefix-Tree (Trie) Autocomplete & Suggestion Engine for NovaSearch.
Provides O(P) prefix lookup where P is prefix length, and top-k completion ranking.
"""

from typing import Dict, List, Optional, Tuple


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.is_terminal: bool = False
        self.weight: float = 0.0


class TrieSuggester:
    """Fast prefix trie for search autocomplete and query suggestion."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._count = 0

    def insert(self, word: str, weight: float = 1.0) -> None:
        """Inserts a word into the trie with an associated popularity weight."""
        word = word.strip().lower()
        if not word:
            return

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_terminal:
            self._count += 1
        node.is_terminal = True
        node.weight = max(node.weight, weight)

    def contains(self, word: str) -> bool:
        """Returns True if the exact word exists in the trie."""
        word = word.strip().lower()
        node = self._find_node(word)
        return node is not None and node.is_terminal

    def suggest(self, prefix: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Returns top_k completions for prefix sorted by weight descending."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []

        node = self._find_node(prefix)
        if node is None:
            return []

        completions: List[Tuple[str, float]] = []
        self._collect(node, list(prefix), completions)
        completions.sort(key=lambda x: x[1], reverse=True)
        return completions[:top_k]

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect(
        self,
        node: TrieNode,
        current_chars: List[str],
        results: List[Tuple[str, float]],
    ) -> None:
        if node.is_terminal:
            results.append(("".join(current_chars), node.weight))

        for char, child in sorted(node.children.items()):
            current_chars.append(char)
            self._collect(child, current_chars, results)
            current_chars.pop()

    @property
    def size(self) -> int:
        return self._count
