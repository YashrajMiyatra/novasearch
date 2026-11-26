"""
Synonym Token Filter Engine for NovaSearch Analysis Pipeline.
Provides Solr/Elasticsearch-style synonym graph expansion and term canonicalization.
"""

from typing import Dict, List, Set


class SynonymFilter:
    """Expands or canonicalizes token streams using term synonym mappings."""

    def __init__(self, mappings: Dict[str, List[str]] = None) -> None:
        self.synonyms: Dict[str, Set[str]] = {}
        if mappings:
            for term, syns in mappings.items():
                self.add_synonyms(term, syns)

    def add_synonyms(self, term: str, syns: List[str], bidirectional: bool = True) -> None:
        """Maps term to synonyms. If bidirectional, all terms map to each other."""
        all_terms = {term.lower()} | {s.lower() for s in syns}
        for t in all_terms:
            self.synonyms.setdefault(t, set()).update(all_terms - {t})

    def expand(self, tokens: List[str]) -> List[str]:
        """Expands token list by injecting synonyms at current positions."""
        expanded: List[str] = []
        for t in tokens:
            low = t.lower()
            expanded.append(low)
            if low in self.synonyms:
                for syn in sorted(self.synonyms[low]):
                    if syn not in expanded:
                        expanded.append(syn)
        return expanded

    def canonicalize(self, token: str, default_to_first: bool = True) -> str:
        """Returns the canonical root form of a token, or the token itself."""
        low = token.lower()
        if low in self.synonyms and default_to_first:
            # Deterministic canonical choice (lexicographically first among group)
            group = sorted(list(self.synonyms[low] | {low}))
            return group[0]
        return low
