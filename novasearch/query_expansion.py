"""
Query Expansion and Synonym Reformulation for NovaSearch.
Improves lexical and hybrid search recall by augmenting queries with domain synonyms.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Set
from .tokenizer import tokenize


class QueryExpander:
    """
    Expands search queries with domain synonyms and weighted reformulation.
    """

    DEFAULT_SYNONYMS: Dict[str, List[str]] = {
        "fast": ["quick", "rapid", "high-performance"],
        "quick": ["fast", "rapid"],
        "search": ["retrieval", "lookup", "query"],
        "database": ["datastore", "store", "storage"],
        "error": ["exception", "failure", "bug"],
        "auth": ["authentication", "authorization", "login"],
        "container": ["sandbox", "isolation", "cgroup"],
        "neural": ["deep-learning", "model", "network"],
    }

    def __init__(self, synonym_map: Optional[Dict[str, List[str]]] = None) -> None:
        self.synonyms = synonym_map or self.DEFAULT_SYNONYMS

    def expand(self, query: str, max_synonyms_per_term: int = 2) -> str:
        """
        Expands the query string by appending synonyms for matched tokens.
        Preserves original query structure.
        """
        tokens = tokenize(query)
        if not tokens:
            return query

        expanded_terms: List[str] = list(tokens)
        seen: Set[str] = set(tokens)

        for t in tokens:
            t_lower = t.lower()
            if t_lower in self.synonyms:
                for syn in self.synonyms[t_lower][:max_synonyms_per_term]:
                    if syn not in seen:
                        expanded_terms.append(syn)
                        seen.add(syn)

        return " ".join(expanded_terms)
