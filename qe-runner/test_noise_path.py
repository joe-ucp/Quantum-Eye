"""Diagnostic script to verify noise model import and application."""

import json
import random
import sys
from typing import Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def _normalize(counts: Dict[str, int]) -> Dict[str, float]:
    total = max(1, sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def _bell_counts(backend: AerSimulator) -> Dict[str, int]:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure_all()
    compiled = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=123)
    result = backend.run(compiled, shots=512, seed_simulator=123).result()
    counts = result.get_counts()
    if isinstance(counts, list):  # pragma: no cover - legacy path
        merged: Dict[str, int] = {}
        for entry in counts:
            merged.update(entry)
        return merged
    return counts  # type: ignore[return-value]


def main() -> None:
    random.seed(123)
    np.random.seed(123)

    try:
        from quantum_eye.adapters.noise_models import get_noise_model_by_type  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(json.dumps({"import": f"failed: {exc}"}))
        sys.exit(1)

    noise_model = get_noise_model_by_type(
        noise_type="combined", noise_level=0.25, metadata={"num_qubits": 2}
    )
    info = {
        "noise_instructions": len(getattr(noise_model, "noise_instructions", [])),
        "basis_gates": getattr(noise_model, "basis_gates", None),
    }

    clean_backend = AerSimulator()
    noisy_backend = AerSimulator(
        noise_model=noise_model, basis_gates=noise_model.basis_gates
    )

    clean = _normalize(_bell_counts(clean_backend))
    noisy = _normalize(_bell_counts(noisy_backend))

    keys = set(clean) | set(noisy)
    l1 = sum(abs(clean.get(k, 0.0) - noisy.get(k, 0.0)) for k in keys)
    info.update(
        {
            "counts_clean": clean,
            "counts_noisy": noisy,
            "l1_diff": l1,
        }
    )

    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

