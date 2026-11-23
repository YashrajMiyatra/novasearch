"""
Metadata Filtering Engine for NovaSearch.
Filters hybrid search candidate documents against structured metadata predicates.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


class MetadataFilter:
    """Evaluates whether document metadata satisfies query filter criteria."""

    @staticmethod
    def matches(metadata: Dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """
        Evaluates a document's metadata against filter_spec.
        Supported operators:
          {"field": "val"}                -> exact equality
          {"field": {"$in": [...]}}       -> set membership
          {"field": {"$gt": val}}         -> greater than
          {"field": {"$gte": val}}        -> greater than or equal
          {"field": {"$lt": val}}         -> less than
          {"field": {"$lte": val}}        -> less than or equal
          {"field": {"$ne": val}}         -> not equal
        """
        if not filter_spec:
            return True

        for key, expected in filter_spec.items():
            if key not in metadata:
                return False
            actual = metadata[key]

            if isinstance(expected, dict):
                for op, target in expected.items():
                    if op == "$eq" and actual != target:
                        return False
                    elif op == "$ne" and actual == target:
                        return False
                    elif op == "$in" and actual not in target:
                        return False
                    elif op == "$nin" and actual in target:
                        return False
                    elif op == "$gt" and not (actual > target):
                        return False
                    elif op == "$gte" and not (actual >= target):
                        return False
                    elif op == "$lt" and not (actual < target):
                        return False
                    elif op == "$lte" and not (actual <= target):
                        return False
            else:
                if actual != expected:
                    return False

        return True
