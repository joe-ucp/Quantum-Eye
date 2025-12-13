from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from constraint_cache import ConstraintCache

# ---- Pauli helpers ---------------------------------------------------------

_PAULI_SINGLE: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _label_for(n_qubits: int, ops: Dict[int, str]) -> str:
    """
    Build a Pauli label string of length n_qubits.

    Qubit index 0 corresponds to the rightmost character of the label (little-endian),
    matching the bitstring convention used in the experiment utilities.
    """
    chars = ["I"] * n_qubits
    for q, pauli in ops.items():
        pos = n_qubits - 1 - q
        chars[pos] = pauli
    return "".join(chars)


def _pauli_matrix(label: str) -> np.ndarray:
    op = np.array([[1.0]], dtype=complex)
    for ch in reversed(label):  # rightmost char acts on qubit 0
        op = np.kron(_PAULI_SINGLE[ch], op)
    return op


_PAULI_CACHE: Dict[str, np.ndarray] = {}


def _pauli_from_cache(label: str) -> np.ndarray:
    if label not in _PAULI_CACHE:
        _PAULI_CACHE[label] = _pauli_matrix(label)
    return _PAULI_CACHE[label]


def expectation_from_state(label: str, state: np.ndarray) -> float:
    op = _pauli_from_cache(label)
    return float(np.real(np.vdot(state, op @ state)))


def expectation_from_counts(label: str, counts: Dict[str, int], shots: int) -> float:
    """
    Expectation of a Z-string from Z-basis counts. Assumes only I/Z in label.
    """
    if "X" in label or "Y" in label:
        raise ValueError("Counts-based expectation only supports I/Z strings.")
    total = max(1, shots)
    exp_val = 0.0
    for bitstring, ct in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 0
        for idx, pauli in enumerate(reversed(label)):
            if pauli != "Z":
                continue
            if bits[-1 - idx] == "1":
                parity ^= 1
        contrib = -1.0 if parity else 1.0
        exp_val += contrib * ct
    return exp_val / total


# ---- Candidate representation ----------------------------------------------


@dataclass
class CandidateHamiltonian:
    structure_id: str
    coeff_id: int
    terms: List[Tuple[str, float]]
    symmetry: Dict[str, Any]
    reachability: Dict[str, Any]

    @property
    def term_labels(self) -> List[str]:
        return [label for (label, _) in self.terms]


# ---- Candidate generation --------------------------------------------------


def _base_structures(n_qubits: int) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Return deterministic base structures with nominal coefficients."""
    diag_terms: List[Tuple[str, float]] = [(_label_for(n_qubits, {}), -1.0)]
    diag_terms += [(_label_for(n_qubits, {q: "Z"}), 0.18 + 0.01 * q) for q in range(n_qubits)]
    diag_terms += [
        (_label_for(n_qubits, {q: "Z", q + 1: "Z"}), -0.08 - 0.01 * (q % 2))
        for q in range(n_qubits - 1)
    ]

    # Add a few off-diagonal variants to make elimination meaningful.
    xx_pairs = diag_terms + [
        (_label_for(n_qubits, {q: "X", q + 1: "X"}), 0.12) for q in range(0, n_qubits - 1, 2)
    ]
    xy_mixed = diag_terms + [
        (_label_for(n_qubits, {q: "X", q + 1: "X"}), 0.10) for q in range(0, n_qubits - 1, 3)
    ]
    xy_mixed += [
        (_label_for(n_qubits, {q: "Y", q + 1: "Y"}), 0.08) for q in range(1, n_qubits - 1, 3)
    ]
    sparse_mix = diag_terms[:-2] + [
        (_label_for(n_qubits, {0: "X", 2: "X"}), 0.05),
        (_label_for(n_qubits, {1: "Y", 3: "Y"}), -0.06),
    ]

    return [
        ("diag_only", diag_terms),
        ("xx_pairs", xx_pairs),
        ("xy_mixed", xy_mixed),
        ("sparse_mix", sparse_mix),
    ]


def generate_candidate_family(n_qubits: int, seed: int, coeff_variants: int = 3) -> List[CandidateHamiltonian]:
    rng = np.random.default_rng(seed)
    family: List[CandidateHamiltonian] = []
    structures = _base_structures(n_qubits)
    for structure_id, terms in structures:
        for coeff_id in range(coeff_variants):
            perturbed_terms: List[Tuple[str, float]] = []
            scale = rng.uniform(0.9, 1.1)
            for label, coeff in terms:
                noise = rng.normal(loc=0.0, scale=0.02 * max(1.0, abs(coeff)))
                perturbed_terms.append((label, float(coeff * scale + noise)))
            offdiag = sum(1 for lbl, _ in perturbed_terms if ("X" in lbl or "Y" in lbl))
            reachability = {
                "depth": 1 + max(0, offdiag // 2),
                "gates": ["ry", "rz", "cz", "cx"],
            }
            symmetry = {"parity": "even"}  # default sector for this synthetic family
            family.append(
                CandidateHamiltonian(
                    structure_id=structure_id,
                    coeff_id=coeff_id,
                    terms=perturbed_terms,
                    symmetry=symmetry,
                    reachability=reachability,
                )
            )
    return family


# ---- Physics helpers -------------------------------------------------------


def build_hamiltonian_matrix(terms: Sequence[Tuple[str, float]], n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for label, coeff in terms:
        H += coeff * _pauli_from_cache(label)
    return H


def ground_state(terms: Sequence[Tuple[str, float]], n_qubits: int) -> Tuple[float, np.ndarray]:
    H = build_hamiltonian_matrix(terms, n_qubits)
    evals, evecs = np.linalg.eigh(H)
    idx = int(np.argmin(np.real(evals)))
    return float(np.real(evals[idx])), np.array(evecs[:, idx]).flatten()


def sample_z_counts(state: np.ndarray, shots: int, n_qubits: int, rng: np.random.Generator) -> Dict[str, int]:
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    outcomes = rng.choice(len(probs), size=shots, p=probs)
    counts: Dict[str, int] = {}
    for idx in outcomes:
        bitstring = format(idx, f"0{n_qubits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def parity_from_counts(counts: Dict[str, int], shots: int) -> str:
    total = max(1, shots)
    odd = 0
    for bitstring, ct in counts.items():
        bits = bitstring.replace(" ", "")
        if bits.count("1") % 2 == 1:
            odd += ct
    odd_prob = odd / total
    if odd_prob < 0.1:
        return "even"
    if odd_prob > 0.9:
        return "odd"
    return "mixed"


def parity_from_state(state: np.ndarray, n_qubits: int) -> str:
    probs = np.abs(state) ** 2
    idxs = np.arange(len(probs))
    # np.frompyfunc keeps Python int.bit_count without dtype issues
    bit_counts = np.frompyfunc(int.bit_count, 1, 1)(idxs).astype(int)
    odd_mask = (bit_counts % 2) == 1
    odd_prob = float(probs[odd_mask].sum())
    if odd_prob < 0.1:
        return "even"
    if odd_prob > 0.9:
        return "odd"
    return "mixed"


def measurement_observables(n_qubits: int) -> List[str]:
    obs: List[str] = []
    obs += [_label_for(n_qubits, {q: "Z"}) for q in range(n_qubits)]
    obs += [_label_for(n_qubits, {q: "Z", q + 1: "Z"}) for q in range(n_qubits - 1)]
    return sorted(set(obs))


def measurement_bounds_from_counts(
    counts: Dict[str, int], observables: List[str], shots: int
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    expectations: Dict[str, float] = {}
    for obs in observables:
        val = expectation_from_counts(obs, counts, shots)
        err = np.sqrt(max(0.0, 1.0 - val**2) / max(1, shots))
        margin = 3.0 * err
        lo = max(-1.0, val - margin)
        hi = min(1.0, val + margin)
        bounds[obs] = (float(lo), float(hi))
        expectations[obs] = float(val)
    return bounds, expectations


def measurement_bounds_from_state(
    state: np.ndarray, observables: List[str], shots: int
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    expectations: Dict[str, float] = {}
    for obs in observables:
        val = expectation_from_state(obs, state)
        err = np.sqrt(max(0.0, 1.0 - val**2) / max(1, shots))
        margin = 3.0 * err
        lo = max(-1.0, val - margin)
        hi = min(1.0, val + margin)
        bounds[obs] = (float(lo), float(hi))
        expectations[obs] = float(val)
    return bounds, expectations


# ---- Elimination logic -----------------------------------------------------


def evaluate_candidate(
    candidate: CandidateHamiltonian,
    observables: List[str],
    measurement_bounds: Dict[str, Tuple[float, float]],
    parity_target: str,
    max_depth: int,
    n_qubits: int,
) -> Tuple[List[str], Dict[str, float], str]:
    energy, state = ground_state(candidate.terms, n_qubits)
    preds = {obs: expectation_from_state(obs, state) for obs in observables}
    reasons: List[str] = []

    for obs, (lo, hi) in measurement_bounds.items():
        val = preds.get(obs, 0.0)
        if val < lo - 1e-9 or val > hi + 1e-9:
            reasons.append(f"measurement_mismatch:{obs}:{val:.3f} not in [{lo:.3f},{hi:.3f}]")

    # Require presence of strongly evidenced terms (interval excludes 0 by a margin).
    for obs, (lo, hi) in measurement_bounds.items():
        if lo > 0.05 or hi < -0.05:
            if obs not in candidate.term_labels:
                reasons.append(f"missing_required_term:{obs}")

    parity = parity_from_state(state, n_qubits)
    if parity_target in ("even", "odd") and parity != parity_target:
        reasons.append(f"parity_mismatch:{parity}!={parity_target}")

    depth = candidate.reachability.get("depth")
    if depth is not None and depth > max_depth:
        reasons.append(f"depth_exceeds:{depth}>{max_depth}")

    return reasons, preds, parity


def eliminate_candidates(
    candidates: List[CandidateHamiltonian],
    observables: List[str],
    measurement_bounds: Dict[str, Tuple[float, float]],
    parity_target: str,
    max_depth: int,
    n_qubits: int,
) -> Tuple[List[CandidateHamiltonian], List[Dict[str, Any]]]:
    survivors: List[CandidateHamiltonian] = []
    eliminated_details: List[Dict[str, Any]] = []

    for cand in candidates:
        reasons, preds, parity = evaluate_candidate(
            cand,
            observables=observables,
            measurement_bounds=measurement_bounds,
            parity_target=parity_target,
            max_depth=max_depth,
            n_qubits=n_qubits,
        )
        if reasons:
            eliminated_details.append(
                {
                    "structure_id": cand.structure_id,
                    "coeff_id": cand.coeff_id,
                    "reasons": reasons,
                    "parity": parity,
                }
            )
        else:
            survivors.append(cand)
    return survivors, eliminated_details


# ---- Experiment orchestration ---------------------------------------------


def run_elimination_experiment(
    n_qubits: int = 6,
    seed: int = 12345,
    shots: int = 4096,
    max_depth: int = 3,
    validation_probes: List[str] | None = None,
    validation_shots: int = 1024,
    coeff_variants: int = 3,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    family = generate_candidate_family(n_qubits=n_qubits, seed=seed, coeff_variants=coeff_variants)

    # Pick a deterministic "true" Hamiltonian that is not purely diagonal if available.
    true_candidate = next((c for c in family if c.structure_id != "diag_only"), family[0])
    true_energy, true_state = ground_state(true_candidate.terms, n_qubits)

    observables = measurement_observables(n_qubits)
    counts = sample_z_counts(true_state, shots=shots, n_qubits=n_qubits, rng=rng)
    measurement_bounds, measured_expectations = measurement_bounds_from_counts(counts, observables, shots)
    measured_parity = parity_from_counts(counts, shots)

    survivors, eliminated_details = eliminate_candidates(
        candidates=family,
        observables=observables,
        measurement_bounds=measurement_bounds,
        parity_target=measured_parity,
        max_depth=max_depth,
        n_qubits=n_qubits,
    )

    # Optional validation probes: add a couple of off-diagonal checks to contradict candidates cheaply.
    if validation_probes is None:
        validation_probes = []
        if n_qubits >= 2:
            validation_probes.append(_label_for(n_qubits, {0: "X", 1: "X"}))
        if n_qubits >= 3:
            validation_probes.append(_label_for(n_qubits, {0: "Y", 2: "Y"}))
        if n_qubits >= 4:
            validation_probes.append(_label_for(n_qubits, {2: "X", 3: "X"}))
        if n_qubits >= 4:
            validation_probes.append(_label_for(n_qubits, {1: "Y", 3: "Y"}))
        if n_qubits >= 5:
            validation_probes.append(_label_for(n_qubits, {3: "X", 4: "X"}))
        if n_qubits >= 6:
            validation_probes.append(_label_for(n_qubits, {4: "X", 5: "X"}))

    validation_bounds: Dict[str, Tuple[float, float]] = {}
    validation_expectations: Dict[str, float] = {}
    validation_eliminated: List[Dict[str, Any]] = []
    if validation_probes:
        vbounds, vexp = measurement_bounds_from_state(true_state, validation_probes, validation_shots)
        validation_bounds.update(vbounds)
        validation_expectations.update(vexp)

        combined_bounds = dict(measurement_bounds)
        combined_bounds.update(validation_bounds)

        survivors_after_probe, eliminated_after_probe = eliminate_candidates(
            candidates=survivors,
            observables=observables + validation_probes,
            measurement_bounds=combined_bounds,
            parity_target=measured_parity,
            max_depth=max_depth,
            n_qubits=n_qubits,
        )
        validation_eliminated = eliminated_after_probe
        survivors = survivors_after_probe
        measurement_bounds = combined_bounds
        measured_expectations.update(validation_expectations)

    allowed_terms: set[str] = set()
    for cand in survivors:
        allowed_terms.update(cand.term_labels)
    eliminated_terms: set[str] = set()
    for cand in family:
        if cand not in survivors:
            eliminated_terms.update(cand.term_labels)
    forbidden_terms = eliminated_terms - allowed_terms

    cache = ConstraintCache(
        term_support=allowed_terms,
        forbidden_terms=forbidden_terms,
        symmetry_sectors={} if measured_parity == "mixed" else {"parity": measured_parity},
        reachability={"max_depth": max_depth, "allowed_gates": ["ry", "rz", "cz", "cx"]},
        measurement_bounds=measurement_bounds,
        metadata={
            "n_qubits": n_qubits,
            "shots": shots,
            "validation_shots": validation_shots if validation_probes else 0,
            "seed": seed,
            "true_structure": true_candidate.structure_id,
            "true_coeff_id": true_candidate.coeff_id,
            "initial_candidates": len(family),
            "survivors": len(survivors),
            "true_energy": true_energy,
            "coeff_variants": coeff_variants,
        },
    )

    experiment = {
        "initial_count": len(family),
        "final_count": len(survivors),
        "measured_parity": measured_parity,
        "measurement_expectations": measured_expectations,
        "measurement_bounds": measurement_bounds,
        "validation_probes": validation_probes,
        "validation_expectations": validation_expectations,
        "coeff_variants": coeff_variants,
        "true_candidate": {
            "structure_id": true_candidate.structure_id,
            "coeff_id": true_candidate.coeff_id,
            "energy": true_energy,
        },
        "survivors": [
            {"structure_id": c.structure_id, "coeff_id": c.coeff_id, "terms": c.terms} for c in survivors
        ],
        "eliminated": eliminated_details,
        "validation_eliminated": validation_eliminated,
    }

    return {
        "cache": cache,
        "experiment": experiment,
    }


if __name__ == "__main__":  # pragma: no cover - manual check
    res = run_elimination_experiment()
    cache_path = Path("constraint_cache.example.json")
    res["cache"].to_json(cache_path)
    print(f"[demo] survivors={res['experiment']['final_count']} / {res['experiment']['initial_count']}")
    print(f"[demo] wrote cache to {cache_path}")

