"""
Adversarial Module 4: Noise × shots regime map.

We sweep:
  noise_level ∈ {0.00, 0.02, 0.05, 0.08, 0.12}
  shots ∈ {128, 256, 512, 1024, 4096}

For each grid point and each fixed seed, we:
- generate a deterministic parameterized 2-qubit circuit (seeded),
- compute its ideal statevector and ideal frequency signature,
- construct deterministic noisy Z counts (noise modeled as mixing with uniform),
- compute the counts-derived frequency signature,
- compute signature overlap and directional observable deltas (ΔX, ΔY) between:
    - a proxy sqrt(p) statevector from Z counts
    - the ideal statevector

Artifacts (deterministic filenames) are written under Quantum-Eye/artifacts/noise_shots_regime_map/.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, SparsePauliOp

# Add parent directory and src to path to allow imports (match existing tests)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from quantum_eye.core.ucp import UCPIdentity
from quantum_eye.core.transform import UcpFrequencyTransform

from tests.circuit_generators import generate_parameterized_circuit
from tests._adversarial_utils import ModuleRunContext, SEEDS, artifact_path, write_csv, write_json


NOISE_LEVELS = [0.00, 0.02, 0.05, 0.08, 0.12]
SHOTS_LIST = [128, 256, 512, 1024, 4096]


def _z_probs_from_statevector(sv: Statevector, n_qubits: int) -> Dict[str, float]:
    probs = sv.probabilities_dict()
    out: Dict[str, float] = {}
    for bitstring, p in probs.items():
        out[str(bitstring).zfill(n_qubits)] = float(p)
    return out


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


def _sqrtp_state_from_counts(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    total = float(sum(counts.values()))
    if total <= 0:
        v = np.zeros(2**n_qubits, dtype=complex)
        v[0] = 1.0
        return v
    v = np.zeros(2**n_qubits, dtype=complex)
    for bitstring, c in counts.items():
        b = str(bitstring).zfill(n_qubits)
        idx = int(b, 2)
        v[idx] = np.sqrt(float(c) / total)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _expect_xy(state: np.ndarray, n_qubits: int) -> Tuple[List[float], List[float]]:
    sv = Statevector(state)

    def pauli(op: str, q: int) -> SparsePauliOp:
        s = ["I"] * n_qubits
        s[n_qubits - 1 - q] = op  # rightmost is qubit0
        return SparsePauliOp("".join(s))

    xs = [float(np.real(sv.expectation_value(pauli("X", q)))) for q in range(n_qubits)]
    ys = [float(np.real(sv.expectation_value(pauli("Y", q)))) for q in range(n_qubits)]
    return xs, ys


def _signature_from_counts(counts: Dict[str, int], n_qubits: int, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi_from_counts(counts, num_qubits=n_qubits)
    return UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)


def _signature_from_state(state: np.ndarray, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi(state)
    return UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)


def _overlap(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    return float(UcpFrequencyTransform({}).frequency_signature_overlap(sig_a, sig_b))


def test_noise_shots_regime_map() -> None:
    module_name = "noise_shots_regime_map"
    config = {
        "n_qubits": 2,
        "depth": 2,
        "theta_base": 1.0,
        "theta_step": 0.07,
        "alpha": 0.5,
        "beta": 0.5,
        "noise_levels": NOISE_LEVELS,
        "shots_list": SHOTS_LIST,
        "seed_list": SEEDS,
        "noise_model": "p'=(1-noise)p + noise*uniform (Z-basis)",
        "proxy_state": "sqrt(p) from counts",
        "metrics": ["signature_overlap", "delta_x_mean_abs", "delta_y_mean_abs"],
    }
    ctx = ModuleRunContext(module_name=module_name, config=config)

    per_point_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []

    for noise in NOISE_LEVELS:
        for shots in SHOTS_LIST:
            overlaps: List[float] = []
            dxs: List[float] = []
            dys: List[float] = []
            for seed in SEEDS:
                theta = float(config["theta_base"] + config["theta_step"] * int(seed))
                qc = generate_parameterized_circuit(
                    n_qubits=config["n_qubits"],
                    depth=config["depth"],
                    theta=theta,
                    seed=int(seed),
                )
                sv = Statevector.from_instruction(qc)
                ideal_state = np.asarray(sv.data)
                ideal_sig = _signature_from_state(ideal_state, alpha=config["alpha"], beta=config["beta"])

                z_probs = _z_probs_from_statevector(sv, n_qubits=config["n_qubits"])
                noisy_probs = _apply_noise_to_probs(z_probs, noise_level=noise, n_qubits=config["n_qubits"])
                counts = _deterministic_counts_from_probs(noisy_probs, shots=shots, n_qubits=config["n_qubits"])
                counts_sig = _signature_from_counts(counts, n_qubits=config["n_qubits"], alpha=config["alpha"], beta=config["beta"])

                o = _overlap(counts_sig, ideal_sig)
                overlaps.append(o)

                proxy_state = _sqrtp_state_from_counts(counts, n_qubits=config["n_qubits"])
                x_true, y_true = _expect_xy(ideal_state, n_qubits=config["n_qubits"])
                x_pred, y_pred = _expect_xy(proxy_state, n_qubits=config["n_qubits"])

                dx = float(np.mean([abs(a - b) for a, b in zip(x_true, x_pred)]))
                dy = float(np.mean([abs(a - b) for a, b in zip(y_true, y_pred)]))
                dxs.append(dx)
                dys.append(dy)

                per_seed_rows.append(
                    {
                        "noise_level": float(noise),
                        "shots": int(shots),
                        "seed": int(seed),
                        "theta": float(theta),
                        "signature_overlap": float(o),
                        "delta_x_mean_abs": float(dx),
                        "delta_y_mean_abs": float(dy),
                    }
                )

            per_point_rows.append(
                {
                    "noise_level": float(noise),
                    "shots": int(shots),
                    "overlap_mean": float(np.mean(overlaps)),
                    "overlap_std": float(np.std(overlaps)),
                    "delta_x_mean_abs": float(np.mean(dxs)),
                    "delta_y_mean_abs": float(np.mean(dys)),
                }
            )

    summary = ctx.base_payload()
    summary.update({"grid": per_point_rows})

    json_out = artifact_path(module_name, f"grid_{ctx.run_id}.json")
    csv_grid_out = artifact_path(module_name, f"grid_{ctx.run_id}.csv")
    csv_seed_out = artifact_path(module_name, f"per_seed_{ctx.run_id}.csv")
    png_out = artifact_path(module_name, f"overlap_heatmap_{ctx.run_id}.png")

    write_json(json_out, summary)
    write_csv(
        csv_grid_out,
        per_point_rows,
        fieldnames=["noise_level", "shots", "overlap_mean", "overlap_std", "delta_x_mean_abs", "delta_y_mean_abs"],
    )
    write_csv(
        csv_seed_out,
        per_seed_rows,
        fieldnames=["noise_level", "shots", "seed", "theta", "signature_overlap", "delta_x_mean_abs", "delta_y_mean_abs"],
    )

    # Plot heatmap of mean overlap
    overlap_matrix = np.zeros((len(NOISE_LEVELS), len(SHOTS_LIST)), dtype=float)
    for r in per_point_rows:
        i = NOISE_LEVELS.index(float(r["noise_level"]))
        j = SHOTS_LIST.index(int(r["shots"]))
        overlap_matrix[i, j] = float(r["overlap_mean"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(overlap_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Mean signature overlap (counts_sig vs ideal_sig)")
    ax.set_xlabel("shots")
    ax.set_ylabel("noise_level")
    ax.set_xticks(range(len(SHOTS_LIST)))
    ax.set_xticklabels([str(s) for s in SHOTS_LIST])
    ax.set_yticks(range(len(NOISE_LEVELS)))
    ax.set_yticklabels([f"{n:.2f}" for n in NOISE_LEVELS])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_out, dpi=160)
    plt.close(fig)

    # Lightweight sanity check: overlap should generally decrease with noise (on average).
    # Not a strict monotonic requirement, but prevents degenerate all-zeros outputs.
    assert any(r["overlap_mean"] > 0.2 for r in per_point_rows), "Overlap too small everywhere; degenerate signature?"



