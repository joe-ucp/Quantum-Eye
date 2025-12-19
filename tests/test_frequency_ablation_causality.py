"""
Adversarial Module 1: Frequency ablation causality.

Implements the protocol section:
  - bin_shuffle
  - phase_scramble (preserve |F|)
  - band_drop (low-pass only vs high-pass only)

This is simulator-only and uses a deterministic counts construction:
we compute exact Z-basis probabilities from a statevector, apply a deterministic
noise-to-uniform mixture, then deterministically convert to counts for a given
shots budget. This keeps artifacts repeatable without requiring simulator seeding.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add parent directory and src to path to allow imports (match existing tests)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from quantum_eye.core.ucp import UCPIdentity
from quantum_eye.core.transform import UcpFrequencyTransform

from tests.circuit_generators import generate_parameterized_circuit
from tests._adversarial_utils import ModuleRunContext, SEEDS, artifact_path, write_csv, write_json


def _z_probs_from_statevector(sv: Statevector, n_qubits: int) -> Dict[str, float]:
    probs = sv.probabilities_dict()
    # Normalize bitstring widths defensively
    out: Dict[str, float] = {}
    for bitstring, p in probs.items():
        out[str(bitstring).zfill(n_qubits)] = float(p)
    return out


def _apply_noise_to_probs(probs: Dict[str, float], noise_level: float, n_qubits: int) -> Dict[str, float]:
    """
    Deterministic noise model for Z-basis probabilities.

    We use a convex mixture with the uniform distribution:
      p' = (1-ε) p + ε / 2**n
    """
    eps = float(noise_level)
    eps = max(0.0, min(1.0, eps))
    uniform = 1.0 / (2**n_qubits)
    return {k: (1.0 - eps) * float(p) + eps * uniform for k, p in probs.items()}


def _deterministic_counts_from_probs(probs: Dict[str, float], shots: int, n_qubits: int) -> Dict[str, int]:
    """
    Deterministically allocate counts by rounding then fixing the residual by
    distributing remaining shots to the largest fractional parts.
    """
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
        order = np.argsort(-frac)  # descending fractional parts
        for idx in order[:residual]:
            base[idx] += 1

    return {k: int(c) for k, c in zip(keys, base) if int(c) > 0}


def _signature_from_counts(counts: Dict[str, int], n_qubits: int, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi_from_counts(counts, num_qubits=n_qubits)
    signature = UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)
    return signature


def _signature_from_state(state: np.ndarray, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi(state)
    signature = UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)
    return signature


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
        perm = rng.permutation(flat.size)
        F = flat[perm].reshape(F.shape)
    elif mode == "phase_scramble":
        mag = np.abs(F)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=F.shape)
        F = mag * np.exp(1j * phase)
    elif mode == "band_drop_low":
        # Keep a low-frequency square around the DC component.
        # NOTE: UcpFrequencyTransform uses FFT conventions where DC is at (0,0).
        h, w = F.shape
        # Aggressive low-pass: keep only a tiny DC patch to destroy scale information.
        r = 1
        mask = np.zeros_like(F, dtype=bool)
        mask[:r, :r] = True
        F = np.where(mask, F, 0.0)
    elif mode == "band_drop_high":
        # Keep only extreme high-frequency corner patches (aggressive high-pass).
        # This is intentionally destructive: preserve some energy while destroying scale info.
        h, w = F.shape
        r = 4
        mask = np.zeros_like(F, dtype=bool)
        # Three corners excluding the DC corner (0,0)
        mask[:r, -r:] = True
        mask[-r:, :r] = True
        mask[-r:, -r:] = True
        F = np.where(mask, F, 0.0)
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    out["full_transform"] = F
    return out


@dataclass(frozen=True)
class _Ref:
    label: str
    circuit: QuantumCircuit
    signature: Dict[str, Any]


def _make_reference_library(alpha: float, beta: float) -> List[_Ref]:
    # Small library of distinct 2-qubit parameterized circuits.
    n_qubits = 2
    depth = 2
    # Avoid near-trivial rotation regimes that can collapse energy into a DC-dominated signature.
    thetas = [0.9, 1.2, 1.5, 1.8]
    refs: List[_Ref] = []
    for i, theta in enumerate(thetas):
        qc = generate_parameterized_circuit(n_qubits=n_qubits, depth=depth, theta=theta, seed=100 + i)
        sv = Statevector.from_instruction(qc)
        sig = _signature_from_state(np.asarray(sv.data), alpha=alpha, beta=beta)
        refs.append(_Ref(label=f"ref_{i}", circuit=qc, signature=sig))
    return refs


def _score_margin(target_sig: Dict[str, Any], refs: List[_Ref], true_label: str) -> Tuple[float, float]:
    overlaps = [(r.label, _overlap(target_sig, r.signature)) for r in refs]
    overlaps.sort(key=lambda x: x[1], reverse=True)
    top_label, top_overlap = overlaps[0]
    true_overlap = dict(overlaps).get(true_label, 0.0)
    best_other = max([o for lbl, o in overlaps if lbl != true_label] or [0.0])
    margin = float(true_overlap - best_other)
    correct = 1.0 if top_label == true_label else 0.0
    return margin, correct


def _true_overlap(target_sig: Dict[str, Any], refs: List[_Ref], true_label: str) -> float:
    ref_sig = next((r.signature for r in refs if r.label == true_label), None)
    if ref_sig is None:
        return 0.0
    return _overlap(target_sig, ref_sig)


def test_frequency_ablation_causality() -> None:
    module_name = "frequency_ablation_causality"
    config = {
        "n_qubits": 2,
        "shots": 2048,
        "noise_level": 0.05,
        "alpha": 0.5,
        "beta": 0.5,
        "reference_library_size": 4,
        "seed_list": SEEDS,
        "metric": "true_overlap_to_correct_reference",
    }
    ctx = ModuleRunContext(module_name=module_name, config=config)

    refs = _make_reference_library(alpha=config["alpha"], beta=config["beta"])

    per_seed_rows: List[Dict[str, Any]] = []
    overlaps_full: List[float] = []
    overlaps_bin: List[float] = []
    overlaps_phase: List[float] = []
    overlaps_low: List[float] = []
    overlaps_high: List[float] = []

    gaps_bin: List[float] = []
    gaps_phase: List[float] = []
    gaps_low: List[float] = []
    gaps_high: List[float] = []

    for seed in SEEDS:
        ref = refs[int(seed) % len(refs)]
        sv = Statevector.from_instruction(ref.circuit)
        z_probs = _z_probs_from_statevector(sv, n_qubits=config["n_qubits"])
        noisy_probs = _apply_noise_to_probs(z_probs, noise_level=config["noise_level"], n_qubits=config["n_qubits"])
        counts = _deterministic_counts_from_probs(noisy_probs, shots=config["shots"], n_qubits=config["n_qubits"])

        full_sig = _signature_from_counts(counts, n_qubits=config["n_qubits"], alpha=config["alpha"], beta=config["beta"])
        bin_sig = _ablated_signature(full_sig, "bin_shuffle", seed=seed)
        phase_sig = _ablated_signature(full_sig, "phase_scramble", seed=seed)
        low_sig = _ablated_signature(full_sig, "band_drop_low", seed=seed)
        high_sig = _ablated_signature(full_sig, "band_drop_high", seed=seed)

        o_full = _true_overlap(full_sig, refs, ref.label)
        o_bin = _true_overlap(bin_sig, refs, ref.label)
        o_phase = _true_overlap(phase_sig, refs, ref.label)
        o_low = _true_overlap(low_sig, refs, ref.label)
        o_high = _true_overlap(high_sig, refs, ref.label)

        overlaps_full.append(o_full)
        overlaps_bin.append(o_bin)
        overlaps_phase.append(o_phase)
        overlaps_low.append(o_low)
        overlaps_high.append(o_high)

        gaps_bin.append(o_full - o_bin)
        gaps_phase.append(o_full - o_phase)
        gaps_low.append(o_full - o_low)
        gaps_high.append(o_full - o_high)

        per_seed_rows.append(
            {
                "seed": seed,
                "true_label": ref.label,
                "true_overlap_full": o_full,
                "true_overlap_bin_shuffle": o_bin,
                "true_overlap_phase_scramble": o_phase,
                "true_overlap_band_drop_low": o_low,
                "true_overlap_band_drop_high": o_high,
                "gap_full_minus_bin_shuffle": o_full - o_bin,
                "gap_full_minus_phase_scramble": o_full - o_phase,
                "gap_full_minus_band_drop_low": o_full - o_low,
                "gap_full_minus_band_drop_high": o_full - o_high,
            }
        )

    summary = ctx.base_payload()
    summary.update(
        {
            "aggregate": {
                "mean_true_overlap_full": float(np.mean(overlaps_full)),
                "mean_true_overlap_bin_shuffle": float(np.mean(overlaps_bin)),
                "mean_true_overlap_phase_scramble": float(np.mean(overlaps_phase)),
                "mean_true_overlap_band_drop_low": float(np.mean(overlaps_low)),
                "mean_true_overlap_band_drop_high": float(np.mean(overlaps_high)),
                "mean_gap_full_minus_bin_shuffle": float(np.mean(gaps_bin)),
                "mean_gap_full_minus_phase_scramble": float(np.mean(gaps_phase)),
                "mean_gap_full_minus_band_drop_low": float(np.mean(gaps_low)),
                "mean_gap_full_minus_band_drop_high": float(np.mean(gaps_high)),
            }
        }
    )

    # Deterministic artifact filenames per module/config
    json_out = artifact_path(module_name, f"summary_{ctx.run_id}.json")
    csv_out = artifact_path(module_name, f"per_seed_{ctx.run_id}.csv")
    write_json(json_out, summary)
    write_csv(
        csv_out,
        per_seed_rows,
        fieldnames=list(per_seed_rows[0].keys()) if per_seed_rows else [],
    )

    # Acceptance checks (protocol recommended thresholds)
    mean_gaps = [
        summary["aggregate"]["mean_gap_full_minus_bin_shuffle"],
        summary["aggregate"]["mean_gap_full_minus_phase_scramble"],
        summary["aggregate"]["mean_gap_full_minus_band_drop_low"],
        summary["aggregate"]["mean_gap_full_minus_band_drop_high"],
    ]
    # Full should beat all ablations on average
    assert all(g > 0.0 for g in mean_gaps), f"Expected full to beat all ablations, got mean gaps: {mean_gaps}"
    # Mean performance gap threshold
    assert float(np.mean(mean_gaps)) >= 0.15, f"Mean gap too small: {float(np.mean(mean_gaps))}"
    # Gap >= 0.10 in >= 80% of seeds, for each ablation
    for name, gaps in [
        ("bin_shuffle", gaps_bin),
        ("phase_scramble", gaps_phase),
        ("band_drop_low", gaps_low),
        ("band_drop_high", gaps_high),
    ]:
        frac = float(np.mean([1.0 if g >= 0.10 else 0.0 for g in gaps]))
        assert frac >= 0.80, f"{name}: gap>=0.10 fraction too low: {frac:.2f}"


