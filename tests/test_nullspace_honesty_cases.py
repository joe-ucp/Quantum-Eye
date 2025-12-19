"""
Adversarial Module 3: Null-space honesty cases.

Goal: explicitly demonstrate where the counts-only method cannot work.

We construct pairs of states with:
- identical Z-basis counts (here: uniform distribution),
- and therefore identical counts-derived invariants used by the method,
- but meaningfully different X/Y structure.

Acceptance:
- frequency signature overlap remains high (>= 0.9),
- any signature-overlap classifier is at chance (~0.5) with low variance across seeds.

This failure is expected: identical Z and ZZ invariants.
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

from tests._adversarial_utils import ModuleRunContext, SEEDS, artifact_path, write_csv, write_json


def _uniform_counts(n_qubits: int, shots: int) -> Dict[str, int]:
    shots = int(shots)
    n = 2**n_qubits
    base = shots // n
    rem = shots - base * n
    out: Dict[str, int] = {}
    for i in range(n):
        b = format(i, f"0{n_qubits}b")
        out[b] = base + (1 if i < rem else 0)
    return {k: v for k, v in out.items() if v > 0}


def _signature_from_counts(counts: Dict[str, int], n_qubits: int, alpha: float, beta: float) -> Dict[str, Any]:
    metrics = UCPIdentity({}).phi_from_counts(counts, num_qubits=n_qubits)
    return UcpFrequencyTransform({}).transform(metrics, alpha=alpha, beta=beta)


def _overlap(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    return float(UcpFrequencyTransform({}).frequency_signature_overlap(sig_a, sig_b))


def _ideal_expectations_xy(state: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """
    Compute ideal X and Y single-qubit expectations from the statevector.
    """
    from qiskit.quantum_info import SparsePauliOp

    sv = Statevector(state)

    def pauli_string(op: str, q: int) -> str:
        # Qiskit Pauli label ordering: rightmost character acts on qubit 0.
        s = ["I"] * n_qubits
        s[n_qubits - 1 - q] = op
        return "".join(s)

    exps: Dict[str, float] = {}
    for q in range(n_qubits):
        x_op = SparsePauliOp(pauli_string("X", q))
        y_op = SparsePauliOp(pauli_string("Y", q))
        exps[f"X{q}"] = float(np.real(sv.expectation_value(x_op)))
        exps[f"Y{q}"] = float(np.real(sv.expectation_value(y_op)))
    return exps


def _predict_label_by_signature(query_sig: Dict[str, Any], ref_sigs: Dict[str, Dict[str, Any]]) -> str:
    # Deterministic argmax. If signatures are identical, this will pick the first label.
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
class _NullCase:
    name: str
    n_qubits: int
    label_a: str
    state_a: np.ndarray
    label_b: str
    state_b: np.ndarray
    note: str


def _case_plus_plus_vs_plus_minus() -> _NullCase:
    # |++> vs |+-> differ in X on qubit 1, but have identical Z distribution (uniform).
    qc_a = QuantumCircuit(2)
    qc_a.h(0)
    qc_a.h(1)
    qc_b = QuantumCircuit(2)
    qc_b.h(0)
    qc_b.h(1)
    qc_b.z(1)  # |+> -> |-> phase flip in X basis
    return _NullCase(
        name="plus_plus_vs_plus_minus",
        n_qubits=2,
        label_a="plus_plus",
        state_a=np.asarray(Statevector.from_instruction(qc_a).data),
        label_b="plus_minus",
        state_b=np.asarray(Statevector.from_instruction(qc_b).data),
        note="Identical Z counts (uniform); differing X expectations.",
    )


def _case_plus_plus_vs_plus_iplus() -> _NullCase:
    # |++> vs |+i,+> differ in Y on qubit 0, but have identical Z distribution (uniform).
    qc_a = QuantumCircuit(2)
    qc_a.h(0)
    qc_a.h(1)
    qc_b = QuantumCircuit(2)
    qc_b.h(0)
    qc_b.s(0)  # |+> -> |+i>
    qc_b.h(1)
    return _NullCase(
        name="plus_plus_vs_plus_i_plus",
        n_qubits=2,
        label_a="plus_plus",
        state_a=np.asarray(Statevector.from_instruction(qc_a).data),
        label_b="plus_i_plus",
        state_b=np.asarray(Statevector.from_instruction(qc_b).data),
        note="Identical Z counts (uniform); differing Y expectations.",
    )


def _run_case(case: _NullCase) -> None:
    module_name = "nullspace_honesty_cases"
    config = {
        "case": case.name,
        "n_qubits": case.n_qubits,
        "shots": 2048,
        "noise_level": 0.0,
        "alpha": 0.5,
        "beta": 0.5,
        "seed_list": SEEDS,
        "notes": case.note,
    }
    ctx = ModuleRunContext(module_name=module_name, config=config)

    # Counts are identical by construction: uniform distribution.
    counts = _uniform_counts(case.n_qubits, shots=config["shots"])
    sig = _signature_from_counts(counts, case.n_qubits, alpha=config["alpha"], beta=config["beta"])

    # Both candidates share the same counts, hence same signature in counts-only path.
    ref_sigs = {case.label_a: sig, case.label_b: sig}

    per_seed_rows: List[Dict[str, Any]] = []
    acc: List[float] = []
    overlap_deltas: List[float] = []

    labels = [case.label_a, case.label_b]
    for seed in SEEDS:
        true_label = labels[int(seed) % 2]
        pred = _predict_label_by_signature(sig, ref_sigs)
        acc.append(1.0 if pred == true_label else 0.0)
        oa = _overlap(sig, ref_sigs[case.label_a])
        ob = _overlap(sig, ref_sigs[case.label_b])
        overlap_deltas.append(float(oa - ob))
        per_seed_rows.append(
            {
                "seed": seed,
                "true_label": true_label,
                "pred": pred,
                "correct": acc[-1],
                "overlap_to_a": float(oa),
                "overlap_to_b": float(ob),
                "overlap_delta_a_minus_b": float(oa - ob),
            }
        )

    # Demonstrate that X/Y structure differs materially (from statevectors).
    exps_a = _ideal_expectations_xy(case.state_a, case.n_qubits)
    exps_b = _ideal_expectations_xy(case.state_b, case.n_qubits)
    diff_l1 = float(sum(abs(exps_a[k] - exps_b.get(k, 0.0)) for k in exps_a.keys()))

    summary = ctx.base_payload()
    summary.update(
        {
            "aggregate": {
                "signature_overlap_a_vs_b": float(_overlap(sig, sig)),
                "classification_accuracy": float(np.mean(acc)),
                "overlap_delta_std": float(np.std(overlap_deltas)),
                "xy_expectations_a": exps_a,
                "xy_expectations_b": exps_b,
                "xy_expectations_l1_diff": diff_l1,
            }
        }
    )

    json_out = artifact_path(module_name, f"summary_{ctx.run_id}.json")
    csv_out = artifact_path(module_name, f"per_seed_{ctx.run_id}.csv")
    write_json(json_out, summary)
    write_csv(csv_out, per_seed_rows, fieldnames=list(per_seed_rows[0].keys()) if per_seed_rows else [])

    # Acceptance:
    assert summary["aggregate"]["signature_overlap_a_vs_b"] >= 0.90
    assert abs(summary["aggregate"]["classification_accuracy"] - 0.50) <= 0.10
    assert summary["aggregate"]["overlap_delta_std"] <= 1e-9
    assert summary["aggregate"]["xy_expectations_l1_diff"] >= 1.0


def test_nullspace_honesty_plus_plus_vs_plus_minus() -> None:
    _run_case(_case_plus_plus_vs_plus_minus())


def test_nullspace_honesty_plus_plus_vs_plus_i_plus() -> None:
    _run_case(_case_plus_plus_vs_plus_iplus())


