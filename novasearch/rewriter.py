"""
Query Rewriting and Acronym Normalization Engine for NovaSearch.
Transforms user queries by expanding acronyms (e.g. k8s -> kubernetes)
and canonicalizing terminology before indexing or retrieval.
"""

import re
from typing import Dict, List, Pattern, Tuple


class QueryRewriter:
    """Canonicalizes and rewrites queries using pattern-action rules."""

    DEFAULT_SYNONYMS = {
        r"\bk8s\b": "kubernetes",
        r"\bml\b": "machine learning",
        r"\bdl\b": "deep learning",
        r"\bnn\b": "neural network",
        r"\bnlp\b": "natural language processing",
        r"\bcv\b": "computer vision",
        r"\bci/cd\b": "continuous integration",
    }

    def __init__(self, custom_rules: Dict[str, str] = None) -> None:
        self.rules: List[Tuple[Pattern, str]] = []
        rules_dict = dict(self.DEFAULT_SYNONYMS)
        if custom_rules:
            rules_dict.update(custom_rules)

        for pat, repl in rules_dict.items():
            self.add_rule(pat, repl)

    def add_rule(self, pattern: str, replacement: str) -> None:
        """Adds a regex transformation rule to the pipeline."""
        compiled = re.compile(pattern, re.IGNORECASE)
        self.rules.append((compiled, replacement))

    def rewrite(self, query: str) -> str:
        """Applies normalization and transformation rules sequentially."""
        if not query:
            return ""

        text = query.strip()
        for compiled, repl in self.rules:
            text = compiled.sub(repl, text)

        # Normalize redundant spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """Rewrites a batch of queries."""
        return [self.rewrite(q) for q in queries]
