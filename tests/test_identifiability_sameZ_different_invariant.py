"""
Adversarial Module 2: Identifiability (same trivial Z statistics, different invariant).

We implement two constructions where:
- Single-qubit Z marginals are identical (and other trivial summaries can be identical),
- But a counts-derivable invariant (e.g., ZZ parity/correlation pattern) differs.

We then validate:
- A baseline that only uses trivial summaries is blind (chance),
- Full counts-derived frequency signature discriminates,
- Frequency ablations (bin_shuffle / phase_scramble) kill the advantage.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

# Add parent directory and src to path to allow imports (match existing tests)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from quantum_eye.core.ucp import UCPIdentity
from quantum_eye.core.transform import UcpFrequencyTransform

from tests._adversarial_utils import ModuleRunContext, SEEDS, artifact_path, write_csv, write_json


def _apply_noise_to_probs(probs: Dict[str, float], noise_level: float, n_qubits: int) -> Dict[str, float]:
    eps = float(noise_level)
    eps = max(0.0, min(1.0, eps))
    uniform = 1.0 / (2**n_qubits)
    return {k: (1.0 - eps) * float(p) + eps * uniform for k, p in probs.items()}


def _deterministic_counts_from_probs(probs: Dict[str, float], shots: int, n_qubits: int) -> Dict[str, int]:
    shots = int(shots)
    keys = sorted({str(k).zfill(n_qubits) for k in probs.keys()})
    p = np.array([float(probs.get(k, 0.0)) for k in keys], dtype=float)
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= 0:
        return {"0" * n_qubits: shots}
    p = p / total

    raw = p * shots
    base = np.floor(raw).astype(int)
    residual = shots - int(base.sum())
    if residual > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in order[:residual]:
            base[idx] += 1
    return {k: int(c) for k, c in zip(keys, base) if int(c) > 0}


def _signature_from_counts(counts: Dict[str, int], n_qubits: int, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi_from_counts(counts, num_qubits=n_qubits)
    return UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)


def _overlap(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    return float(UcpFrequencyTransform({}).frequency_signature_overlap(sig_a, sig_b))


def _ablated_signature(sig: Dict[str, Any], mode: str, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    out = dict(sig)
    F = sig.get("full_transform")
    if F is None:
        return out
    F = np.array(F, dtype=np.complex128, copy=True)

    if mode == "bin_shuffle":
        flat = F.flatten()
        F = flat[rng.permutation(flat.size)].reshape(F.shape)
    elif mode == "phase_scramble":
        mag = np.abs(F)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=F.shape)
        F = mag * np.exp(1j * phase)
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    out["full_transform"] = F
    return out


def _single_qubit_marginals(counts: Dict[str, int], n_qubits: int) -> List[float]:
    """
    Trivial Z summary: P(bit=1) per qubit, derived from counts.
    """
    total = float(sum(counts.values())) or 1.0
    p1 = [0.0 for _ in range(n_qubits)]
    for bitstring, c in counts.items():
        b = str(bitstring).zfill(n_qubits)
        for q in range(n_qubits):
            if b[q] == "1":
                p1[q] += float(c) / total
    return [float(x) for x in p1]


def _baseline_predict_label_by_marginals(
    query_counts: Dict[str, int],
    refs: Dict[str, Dict[str, int]],
    n_qubits: int,
) -> str:
    q = np.array(_single_qubit_marginals(query_counts, n_qubits), dtype=float)
    best_label = None
    best_dist = float("inf")
    for label, ref_counts in refs.items():
        r = np.array(_single_qubit_marginals(ref_counts, n_qubits), dtype=float)
        dist = float(np.sum(np.abs(q - r)))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    assert best_label is not None
    return best_label


def _predict_label_by_signature(query_sig: Dict[str, Any], ref_sigs: Dict[str, Dict[str, Any]]) -> str:
    best_label = None
    best_overlap = -1.0
    for label, sig in ref_sigs.items():
        o = _overlap(query_sig, sig)
        if o > best_overlap:
            best_overlap = o
            best_label = label
    assert best_label is not None
    return best_label


@dataclass(frozen=True)
class _Construction:
    name: str
    n_qubits: int
    # two Z-basis distributions (probs) + labels
    label_a: str
    probs_a: Dict[str, float]
    label_b: str
    probs_b: Dict[str, float]
    note: str


def _bell_family_construction() -> _Construction:
    # |Φ+> => Z outcomes 00/11 each 0.5
    # |Ψ+> => Z outcomes 01/10 each 0.5
    # Both have identical single-qubit marginals: P(q=1)=0.5 for each qubit.
    return _Construction(
        name="bell_phi_plus_vs_psi_plus",
        n_qubits=2,
        label_a="phi_plus",
        probs_a={"00": 0.5, "11": 0.5},
        label_b="psi_plus",
        probs_b={"01": 0.5, "10": 0.5},
        note="Same single-qubit Z marginals; differing ZZ parity/correlation pattern.",
    )


def _three_qubit_pair_construction() -> _Construction:
    # Two 2-outcome distributions with identical single-qubit marginals (all 0.5),
    # but different correlation structure.
    # A: support on 000 and 111
    # B: support on 011 and 100
    return _Construction(
        name="three_qubit_ghz_like_parity_swap",
        n_qubits=3,
        label_a="ghz_like_000_111",
        probs_a={"000": 0.5, "111": 0.5},
        label_b="ghz_like_011_100",
        probs_b={"011": 0.5, "100": 0.5},
        note="Same single-qubit Z marginals; differing multi-qubit correlation topology.",
    )


def _run_construction(con: _Construction) -> None:
    module_name = "identifiability_sameZ_different_invariant"
    config = {
        "construction": con.name,
        "n_qubits": con.n_qubits,
        "shots": 2048,
        "noise_level": 0.05,
        "alpha": 0.5,
        "beta": 0.5,
        "seed_list": SEEDS,
        "baseline": "single_qubit_marginals_L1",
        "notes": con.note,
    }
    ctx = ModuleRunContext(module_name=module_name, config=config)

    # Build deterministic reference counts (noise-free) for baseline
    ref_counts = {
        con.label_a: _deterministic_counts_from_probs(con.probs_a, shots=config["shots"], n_qubits=con.n_qubits),
        con.label_b: _deterministic_counts_from_probs(con.probs_b, shots=config["shots"], n_qubits=con.n_qubits),
    }
    # Reference signatures use the exact distributions (as if infinite shots)
    ref_sigs = {
        con.label_a: _signature_from_counts(ref_counts[con.label_a], con.n_qubits, config["alpha"], config["beta"]),
        con.label_b: _signature_from_counts(ref_counts[con.label_b], con.n_qubits, config["alpha"], config["beta"]),
    }

    per_seed_rows: List[Dict[str, Any]] = []
    baseline_acc: List[float] = []
    full_acc: List[float] = []
    ablated_bin_acc: List[float] = []
    ablated_phase_acc: List[float] = []

    # Alternate which label is the query across seeds.
    labels = [con.label_a, con.label_b]
    probs_by_label = {con.label_a: con.probs_a, con.label_b: con.probs_b}

    for seed in SEEDS:
        true_label = labels[int(seed) % 2]
        probs = probs_by_label[true_label]
        noisy_probs = _apply_noise_to_probs(probs, noise_level=config["noise_level"], n_qubits=con.n_qubits)
        counts = _deterministic_counts_from_probs(noisy_probs, shots=config["shots"], n_qubits=con.n_qubits)

        # Sanity: trivial marginals should match (construction intent)
        marg_a = _single_qubit_marginals(ref_counts[con.label_a], con.n_qubits)
        marg_b = _single_qubit_marginals(ref_counts[con.label_b], con.n_qubits)
        assert np.allclose(marg_a, marg_b, atol=1e-9), "Construction violated: marginals must match"

        baseline_pred = _baseline_predict_label_by_marginals(counts, ref_counts, con.n_qubits)
        baseline_acc.append(1.0 if baseline_pred == true_label else 0.0)

        full_sig = _signature_from_counts(counts, con.n_qubits, config["alpha"], config["beta"])
        full_pred = _predict_label_by_signature(full_sig, ref_sigs)
        full_acc.append(1.0 if full_pred == true_label else 0.0)

        bin_sig = _ablated_signature(full_sig, "bin_shuffle", seed=seed)
        bin_pred = _predict_label_by_signature(bin_sig, ref_sigs)
        ablated_bin_acc.append(1.0 if bin_pred == true_label else 0.0)

        phase_sig = _ablated_signature(full_sig, "phase_scramble", seed=seed)
        phase_pred = _predict_label_by_signature(phase_sig, ref_sigs)
        ablated_phase_acc.append(1.0 if phase_pred == true_label else 0.0)

        per_seed_rows.append(
            {
                "seed": seed,
                "true_label": true_label,
                "baseline_pred": baseline_pred,
                "full_pred": full_pred,
                "bin_shuffle_pred": bin_pred,
                "phase_scramble_pred": phase_pred,
                "baseline_correct": 1.0 if baseline_pred == true_label else 0.0,
                "full_correct": 1.0 if full_pred == true_label else 0.0,
                "bin_shuffle_correct": 1.0 if bin_pred == true_label else 0.0,
                "phase_scramble_correct": 1.0 if phase_pred == true_label else 0.0,
            }
        )

    summary = ctx.base_payload()
    summary.update(
        {
            "aggregate": {
                "baseline_accuracy": float(np.mean(baseline_acc)),
                "full_accuracy": float(np.mean(full_acc)),
                "bin_shuffle_accuracy": float(np.mean(ablated_bin_acc)),
                "phase_scramble_accuracy": float(np.mean(ablated_phase_acc)),
            }
        }
    )

    json_out = artifact_path(module_name, f"summary_{ctx.run_id}.json")
    csv_out = artifact_path(module_name, f"per_seed_{ctx.run_id}.csv")
    write_json(json_out, summary)
    write_csv(csv_out, per_seed_rows, fieldnames=list(per_seed_rows[0].keys()) if per_seed_rows else [])

    # Acceptance:
    # - Baseline blind: near chance (<= 0.6 for 2-way)
    # - Full discriminates strongly (>= 0.9)
    # - Ablations kill advantage (<= 0.6)
    assert summary["aggregate"]["baseline_accuracy"] <= 0.60
    assert summary["aggregate"]["full_accuracy"] >= 0.90
    assert summary["aggregate"]["bin_shuffle_accuracy"] <= 0.60
    assert summary["aggregate"]["phase_scramble_accuracy"] <= 0.60


def test_identifiability_bell_family_same_marginals_different_zz() -> None:
    _run_construction(_bell_family_construction())


def test_identifiability_three_qubit_same_marginals_different_topology() -> None:
    _run_construction(_three_qubit_pair_construction())



