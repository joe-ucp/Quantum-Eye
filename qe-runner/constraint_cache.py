from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def _set_to_sorted_list(items: Iterable[str]) -> List[str]:
    return sorted(set(items))


def _list_to_set(items: Iterable[str] | None) -> Set[str]:
    return set(items or [])


@dataclass
class ConstraintCache:
    """
    Constraint cache that records impossibilities from a Hamiltonian-learning run.

    The cache stores what is forbidden or constrained, not point estimates.
    - term_support: Pauli strings allowed by the validated structure.
    - forbidden_terms: Pauli strings ruled out by elimination.
    - symmetry_sectors: Required symmetry labels (e.g., parity, particle number).
    - reachability: Limits from ansatz/gate set (e.g., max_depth, allowed_gates).
    - measurement_bounds: Intervals for expectation values <P> consistent with data.
    """

    term_support: Set[str] = field(default_factory=set)
    forbidden_terms: Set[str] = field(default_factory=set)
    symmetry_sectors: Dict[str, Any] = field(default_factory=dict)
    reachability: Dict[str, Any] = field(default_factory=dict)
    measurement_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "term_support": _set_to_sorted_list(self.term_support),
            "forbidden_terms": _set_to_sorted_list(self.forbidden_terms),
            "symmetry_sectors": self.symmetry_sectors,
            "reachability": self.reachability,
            "measurement_bounds": {
                term: [float(lo), float(hi)] for term, (lo, hi) in self.measurement_bounds.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintCache":
        """Rehydrate from a dict."""
        return cls(
            term_support=_list_to_set(data.get("term_support")),
            forbidden_terms=_list_to_set(data.get("forbidden_terms")),
            symmetry_sectors=data.get("symmetry_sectors", {}) or {},
            reachability=data.get("reachability", {}) or {},
            measurement_bounds={
                term: (float(bounds[0]), float(bounds[1]))
                for term, bounds in (data.get("measurement_bounds") or {}).items()
            },
            metadata=data.get("metadata", {}) or {},
        )

    def to_json(self, path: Path) -> None:
        """Write cache to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ConstraintCache":
        """Load cache from JSON file."""
        return cls.from_dict(json.loads(path.read_text()))

    # ---- validation helpers ----
    def _term_violations(self, candidate_terms: Iterable[str]) -> List[str]:
        violations: List[str] = []
        terms_set = set(candidate_terms)
        # Forbidden terms
        bad_forbidden = terms_set & self.forbidden_terms
        if bad_forbidden:
            violations.append(f"contains_forbidden_terms:{sorted(bad_forbidden)}")
        # Terms outside allowed support (if support is specified)
        if self.term_support:
            outside = terms_set - self.term_support
            if outside:
                violations.append(f"outside_term_support:{sorted(outside)}")
        return violations

    def _symmetry_violations(self, candidate_symmetry: Dict[str, Any] | None) -> List[str]:
        if not self.symmetry_sectors:
            return []
        if candidate_symmetry is None:
            return ["missing_symmetry_info"]
        bad: List[str] = []
        for key, required_val in self.symmetry_sectors.items():
            if key not in candidate_symmetry:
                bad.append(f"missing_symmetry:{key}")
                continue
            if candidate_symmetry[key] != required_val:
                bad.append(f"symmetry_mismatch:{key}:{candidate_symmetry[key]}!= {required_val}")
        return bad

    def _reachability_violations(self, candidate_reach: Dict[str, Any] | None) -> List[str]:
        if not self.reachability:
            return []
        if candidate_reach is None:
            return ["missing_reachability_info"]
        bad: List[str] = []
        # max_depth: candidate depth must not exceed cached max_depth
        max_depth = self.reachability.get("max_depth")
        if max_depth is not None:
            cand_depth = candidate_reach.get("depth")
            if cand_depth is None:
                bad.append("missing_reachability:depth")
            elif float(cand_depth) > float(max_depth):
                bad.append(f"depth_exceeds:{cand_depth}>{max_depth}")
        # allowed_gates: candidate gates must be subset of allowed_gates
        allowed_gates = self.reachability.get("allowed_gates")
        if allowed_gates is not None:
            allowed_set = set(allowed_gates)
            cand_gates = set(candidate_reach.get("gates", []))
            extra = cand_gates - allowed_set
            if extra:
                bad.append(f"gate_not_allowed:{sorted(extra)}")
        return bad

    def _measurement_violations(self, expectations: Dict[str, float] | None) -> List[str]:
        if not self.measurement_bounds or expectations is None:
            return []
        bad: List[str] = []
        for term, bounds in self.measurement_bounds.items():
            if term not in expectations:
                bad.append(f"missing_expectation:{term}")
                continue
            lo, hi = bounds
            val = expectations[term]
            if val < lo - 1e-9 or val > hi + 1e-9:
                bad.append(f"expectation_out_of_bounds:{term}:{val} not in [{lo},{hi}]")
        return bad

    def violations(
        self,
        candidate_terms: Iterable[str],
        *,
        candidate_symmetry: Dict[str, Any] | None = None,
        candidate_reachability: Dict[str, Any] | None = None,
        expectations: Dict[str, float] | None = None,
    ) -> List[str]:
        """Return list of violation reasons for a candidate."""
        reasons: List[str] = []
        reasons.extend(self._term_violations(candidate_terms))
        reasons.extend(self._symmetry_violations(candidate_symmetry))
        reasons.extend(self._reachability_violations(candidate_reachability))
        reasons.extend(self._measurement_violations(expectations))
        return reasons

    def is_candidate_allowed(
        self,
        candidate_terms: Iterable[str],
        *,
        candidate_symmetry: Dict[str, Any] | None = None,
        candidate_reachability: Dict[str, Any] | None = None,
        expectations: Dict[str, float] | None = None,
    ) -> bool:
        """True if candidate passes all cached constraints."""
        return len(
            self.violations(
                candidate_terms,
                candidate_symmetry=candidate_symmetry,
                candidate_reachability=candidate_reachability,
                expectations=expectations,
            )
        ) == 0


