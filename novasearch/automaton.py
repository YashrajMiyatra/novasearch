"""
Levenshtein Automaton Subsystem for NovaSearch.
Implements non-deterministic finite automaton for fast O(N) fuzzy string matching
and dictionary candidate filtering within edit distance k.
"""

from typing import List, Set, Tuple


class LevenshteinAutomaton:
    """Non-deterministic finite state automaton recognizing strings within max edit distance."""

    def __init__(self, target: str, max_edits: int = 2) -> None:
        if max_edits < 0:
            raise ValueError("max_edits must be non-negative")
        self.target = target
        self.max_edits = int(max_edits)

    def start_state(self) -> Tuple[Tuple[int, int], ...]:
        """Returns the initial state with epsilon-transitions for leading deletions."""
        closure: Set[Tuple[int, int]] = {(0, 0)}
        for i in range(1, min(self.max_edits + 1, len(self.target) + 1)):
            closure.add((i, i))
        return tuple(sorted(closure))

    def step(self, state: Tuple[Tuple[int, int], ...], char: str) -> Tuple[Tuple[int, int], ...]:
        """Computes the set of reachable states after consuming input character char."""
        new_state: Set[Tuple[int, int]] = set()
        n = len(self.target)

        for i, e in state:
            # Exact match transition
            if i < n and self.target[i] == char:
                new_state.add((i + 1, e))

            if e < self.max_edits:
                # Insertion error (consume char without advancing target)
                new_state.add((i, e + 1))
                # Substitution error (consume char and advance target)
                if i < n:
                    new_state.add((i + 1, e + 1))

        # Epsilon closure (deletion errors in target without consuming input)
        changed = True
        while changed:
            changed = False
            for i, e in list(new_state):
                if e < self.max_edits and i < n:
                    succ = (i + 1, e + 1)
                    if succ not in new_state:
                        new_state.add(succ)
                        changed = True

        return tuple(sorted(new_state))

    def is_match(self, state: Tuple[Tuple[int, int], ...]) -> bool:
        """Determines if the state contains an accepting sub-state."""
        n = len(self.target)
        return any(i == n for i, _ in state)

    def can_match(self, state: Tuple[Tuple[int, int], ...]) -> bool:
        """Determines if the state can still potentially reach an accepting state."""
        return len(state) > 0

    def matches(self, candidate: str) -> bool:
        """Evaluates whether candidate string is accepted within max edit distance."""
        state = self.start_state()
        for ch in candidate:
            state = self.step(state, ch)
            if not state:
                return False
        return self.is_match(state)

    def filter_candidates(self, candidates: List[str]) -> List[str]:
        """Filters a collection of candidate words matching within max edit distance."""
        return [c for c in candidates if self.matches(c)]
