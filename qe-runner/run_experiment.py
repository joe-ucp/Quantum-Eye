#!/usr/bin/env python
"""
Headless experiment runner for baseline vs adapter in ffsim.

Outputs live in /home/jovyan/persistent-volume when used with compose.test.yaml.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

HOME = Path("/home/jovyan")
PERSISTENT = HOME / "persistent-volume"
DEFAULT_ADAPTER_PATH = PERSISTENT / "adapter"
FFSIM_SRC = HOME / ".src" / "ffsim"
MARKER = PERSISTENT / ".deps-installed.txt"
PINNED_PACKAGES = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "matplotlib==3.8.4",
    "psutil==6.0.0",
    "ffsim==0.0.63",
    "qiskit==1.3.2",
    "qiskit-aer==0.14.2",
]


def log(msg: str) -> None:
    """Lightweight, flush-on-write logger for progress feedback."""
    print(msg, flush=True)


def install_deps() -> None:
    """Install pinned deps once, with optional extras, tracked by marker."""
    PERSISTENT.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "packages": PINNED_PACKAGES,
        "extra": os.environ.get("FFSIM_PIP_EXTRA", "").strip(),
    }
    if marker_payload["extra"]:
        extra_parts = marker_payload["extra"].split()
    else:
        extra_parts = []

    def _deps_present() -> bool:
        try:
            for pkg in ("numpy", "scipy", "matplotlib", "psutil", "ffsim", "qiskit", "qiskit_aer"):
                if importlib.util.find_spec(pkg) is None:
                    return False
            return True
        except Exception:
            return False

    if MARKER.exists():
        try:
            recorded = json.loads(MARKER.read_text())
            if recorded == marker_payload and _deps_present():
                log("[deps] pinned deps already installed; skipping")
                return
        except Exception:
            pass

    log("[deps] installing pinned deps...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        *PINNED_PACKAGES,
        *extra_parts,
    ]
    subprocess.check_call(cmd)
    MARKER.write_text(json.dumps(marker_payload))


def load_modules() -> None:
    """Import modules after deps are installed."""
    global np
    global ffsim
    global DiagonalCoulombHamiltonian, random_real_symmetric_matrix, random_special_orthogonal
    global hartree_fock_state, slater_determinant, random_occupied_orbitals
    # bell workload uses pure numpy; no qiskit imports needed

    import numpy as np  # type: ignore

    import ffsim  # type: ignore
    from ffsim.hamiltonians.diagonal_coulomb_hamiltonian import (
        DiagonalCoulombHamiltonian,
    )
    from ffsim.random import random_real_symmetric_matrix, random_special_orthogonal
    from ffsim.states.slater import hartree_fock_state, slater_determinant
    from ffsim.testing.testing import random_occupied_orbitals

    globals().update(
        {
            "np": np,
            "ffsim": ffsim,
            "DiagonalCoulombHamiltonian": DiagonalCoulombHamiltonian,
            "random_real_symmetric_matrix": random_real_symmetric_matrix,
            "random_special_orthogonal": random_special_orthogonal,
            "hartree_fock_state": hartree_fock_state,
            "slater_determinant": slater_determinant,
            "random_occupied_orbitals": random_occupied_orbitals,
        }
    )


def set_deterministic_seed(seed: int) -> None:
    """Apply deterministic seeding across stdlib, numpy, and JAX if present."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import jax  # type: ignore

        jax.random.key(seed)  # ensure key generation; unused but initializes backend
    except Exception:
        pass


def get_git_sha(repo_path: Path) -> str:
    if not repo_path.exists():
        return "unavailable (source tree not present)"
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def get_image_digest() -> str:
    # Best-effort: prefer env hints; fall back to unknown.
    for key in ("IMAGE_DIGEST", "IMAGE_ID", "DOCKER_IMAGE_DIGEST"):
        if os.environ.get(key):
            return os.environ[key]
    return "unknown"


def get_base_image() -> str:
    return os.environ.get("BASE_IMAGE", "quay.io/jupyter/minimal-notebook:python-3.11")


def get_pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"unavailable ({exc})"]


def get_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory_total_gb"] = round(vm.total / (1024**3), 2)
        info["memory_available_gb"] = round(vm.available / (1024**3), 2)
        info["cpu_freq_mhz"] = getattr(psutil.cpu_freq(), "current", None)
    except Exception as exc:  # pragma: no cover - psutil may be absent
        info["psutil"] = f"unavailable ({exc})"
    return info


def maybe_install_adapter(adapter_path: Path) -> dict[str, Any]:
    """Optionally install the adapter in editable mode."""
    result: dict[str, Any] = {
        "requested_path": str(adapter_path),
        "installed": False,
        "error": None,
    }
    if not adapter_path.exists():
        result["error"] = "adapter path missing"
        return result
    try:
        log(f"[adapter] Installing editable package from {adapter_path}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(adapter_path)]
        )
        result["installed"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def current_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def build_hamiltonian(rng: np.random.Generator, norb: int) -> DiagonalCoulombHamiltonian:
    one_body = random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb = np.stack(
        [
            random_real_symmetric_matrix(norb, seed=rng).astype(float),
            random_real_symmetric_matrix(norb, seed=rng).astype(float),
        ]
    )
    constant = float(rng.standard_normal())
    return DiagonalCoulombHamiltonian(one_body, diag_coulomb, constant=constant)


def simulate_run(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    norb = 4
    nelec: Tuple[int, int] = (2, 2)

    occ = random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rot = random_special_orthogonal(norb, seed=rng)
    ham = build_hamiltonian(rng, norb=norb)
    linop = ham._linear_operator_(norb, nelec)  # type: ignore[attr-defined]

    hf_state = hartree_fock_state(norb, nelec)
    state = slater_determinant(norb, occ, orbital_rotation=orbital_rot)

    t0 = time.perf_counter()
    energy = float(np.real_if_close(state.conj() @ (linop @ state)))
    ref_energy = float(np.real_if_close(hf_state.conj() @ (linop @ hf_state)))
    wall_clock = time.perf_counter() - t0

    fidelity = float(np.abs(np.vdot(hf_state, state)) ** 2)

    mem_mb = current_memory_mb()

    metrics = {
        "energy": energy,
        "reference_energy": ref_energy,
        "energy_error": energy - ref_energy,
        "fidelity_to_reference": fidelity,
        "wall_clock_s": wall_clock,
        "shots_used": 0,
        "max_rss_mb": mem_mb,
        "success": True,
    }

    config = {
        "seed": seed,
        "norb": norb,
        "nelec": nelec,
        "occupied_orbitals": occ,
    }
    return {"metrics": metrics, "config": config}


def counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def shannon_entropy(probs: Dict[str, float]) -> float:
    vals = [p for p in probs.values() if p > 0]
    return -float(sum(p * np.log(p) for p in vals))


def bell_qsv_from_z(probs_z: Dict[str, float]) -> Dict[str, float]:
    p00 = probs_z.get("00", 0.0)
    p01 = probs_z.get("01", 0.0)
    p10 = probs_z.get("10", 0.0)
    p11 = probs_z.get("11", 0.0)
    h = shannon_entropy(probs_z)
    ipr = sum(v * v for v in probs_z.values())
    corr = max(p00 + p11, p01 + p10)
    P = corr * (1 - h / np.log(4))
    S = np.exp(-abs(ipr - 2)) * max(probs_z.values() or [0.0])
    E = np.exp(-abs((h - np.log(2)) / 0.3) ** 2)
    Q = 1.2 * corr
    score = P * S * E * Q
    return {"P": float(P), "S": float(S), "E": float(E), "Q": float(Q), "score": float(score)}


def run_bell(mode: str, seed: int, shots: int = 4096) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    # Ideal Bell probabilities
    z_probs = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}
    ideal_x = {"00": 0.5, "11": 0.5}
    ideal_y = {"01": 0.5, "10": 0.5}

    def sample_counts(probs: Dict[str, float]) -> Dict[str, int]:
        keys = list(probs.keys())
        vec = [probs[k] for k in keys]
        counts_vec = rng.multinomial(shots, vec)
        return {k: int(v) for k, v in zip(keys, counts_vec) if v > 0}

    z_counts = sample_counts(z_probs)
    x_counts = sample_counts(ideal_x)
    y_counts = sample_counts(ideal_y)
    wall = time.perf_counter() - t0

    x_meas = counts_to_probs(x_counts)
    y_meas = counts_to_probs(y_counts)

    def fidelity(p_pred: Dict[str, float], p_meas: Dict[str, float]) -> float:
        all_keys = set(p_pred.keys()) | set(p_meas.keys())
        val = sum(np.sqrt(p_pred.get(k, 0.0) * p_meas.get(k, 0.0)) for k in all_keys)
        return float(val * val)

    x_accuracy = fidelity(ideal_x, x_meas)
    y_accuracy = fidelity(ideal_y, y_meas)
    x_corr = x_meas.get("00", 0.0) + x_meas.get("11", 0.0)
    y_corr = y_meas.get("01", 0.0) + y_meas.get("10", 0.0)

    qsv = bell_qsv_from_z(z_probs)

    metrics = {
        "energy": None,
        "reference_energy": None,
        "energy_error": None,
        "fidelity_to_reference": None,
        "wall_clock_s": wall,
        "shots_used": shots,
        "max_rss_mb": current_memory_mb(),
        "iterations": None,
        "x_accuracy": x_accuracy,
        "y_accuracy": y_accuracy,
        "x_correlation": x_corr,
        "y_correlation": y_corr,
        "qsv_score": qsv["score"],
        "qsv_P": qsv["P"],
        "qsv_S": qsv["S"],
        "qsv_E": qsv["E"],
        "qsv_Q": qsv["Q"],
        "success": True,
    }

    config = {"seed": seed, "shots": shots, "mode": mode}
    return {"metrics": metrics, "config": config}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}

    keys = set()
    for r in runs:
        keys.update(r.get("metrics", {}).keys())

    def stats(key: str) -> dict[str, float]:
        values = [r["metrics"].get(key) for r in runs if r["metrics"].get(key) is not None]
        if not values:
            return {"mean": float("nan"), "std": float("nan")}
        arr = np.array(values, dtype=float)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    return {k: stats(k) for k in sorted(keys)}


def persist_csv(runs: list[dict[str, Any]], path: Path) -> None:
    if not runs:
        return
    fieldnames = [
        "run_id",
        "seed",
        "energy",
        "reference_energy",
        "energy_error",
        "best_energy",
        "best_energy_error",
        "best_iteration",
        "fidelity_to_reference",
        "wall_clock_s",
        "shots_used",
        "total_shots",
        "max_rss_mb",
        "iterations",
        "x_accuracy",
        "y_accuracy",
        "x_correlation",
        "y_correlation",
        "qsv_score",
        "qsv_P",
        "qsv_S",
        "qsv_E",
        "qsv_Q",
        "success",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, run in enumerate(runs):
            row = {
                "run_id": idx,
                "seed": run["config"]["seed"],
            }
            row.update(run["metrics"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ffsim baseline vs adapter experiment")
    parser.add_argument("--mode", choices=["baseline", "adapter"], required=True)
    parser.add_argument("--runs", type=int, default=3, help="Number of seeded runs")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to adapter package to install in adapter mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JSON/CSV outputs (default: Quantum-eye-test/<timestamp>)",
    )
    parser.add_argument(
        "--workload",
        choices=["energy", "bell", "vqe", "hamiltonian_elimination", "cache_reuse", "scaled_cache"],
        default="energy",
        help="Select workload: energy (default), bell (Quantum Eye), vqe stub, hamiltonian_elimination, cache_reuse, or scaled_cache",
    )
    parser.add_argument(
        "--shots-per-eval",
        type=int,
        default=2048,
        help="Shots per energy evaluation (used by VQE workload)",
    )
    parser.add_argument(
        "--vqe-iterations",
        type=int,
        default=40,
        help="SPSA iterations for VQE workload",
    )
    parser.add_argument(
        "--qe-smoothing-window",
        type=float,
        default=45.0,
        help="Gaussian window strength for Quantum Eye smoothing (adapter only)",
    )
    parser.add_argument(
        "--qe-uniform-floor",
        type=float,
        default=0.1,
        help="Uniform mixing floor for Quantum Eye smoothing (adapter only)",
    )
    parser.add_argument(
        "--spsa-a0",
        type=float,
        default=None,
        help="Override SPSA a0 (default depends on mode)",
    )
    parser.add_argument(
        "--spsa-c0",
        type=float,
        default=None,
        help="Override SPSA c0 (default depends on mode)",
    )
    parser.add_argument(
        "--vqe-estimator",
        choices=["qe", "psqe", "psqe_fft", "baseline", "full_baseline"],
        default="qe",
        help="Energy estimator for VQE: qe/psqe/psqe_fft (adapter), baseline (diagonal), or full_baseline (grouped Pauli)",
    )
    parser.add_argument(
        "--noise-type",
        choices=["depolarizing", "amplitude_damping", "phase", "combined", "reset_noise"],
        default=None,
        help="Optional noise model type for simulator",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Noise strength (0 for noiseless). Typical small values: 0.01-0.05",
    )
    parser.add_argument(
        "--elimination-qubits",
        type=int,
        default=6,
        help="Number of qubits for Hamiltonian elimination workload",
    )
    parser.add_argument(
        "--elimination-shots",
        type=int,
        default=4096,
        help="Shots for Z-basis measurements in elimination workload",
    )
    parser.add_argument(
        "--elimination-coeff-variants",
        type=int,
        default=3,
        help="Number of coefficient variants per structure for elimination",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to constraint cache JSON for cache_reuse workload",
    )
    parser.add_argument(
        "--reuse-qubits",
        type=int,
        default=None,
        help="Optional override for number of qubits in cache_reuse workload",
    )
    parser.add_argument(
        "--scaled-seeds",
        type=int,
        default=5,
        help="Number of seeds for scaled_cache workload",
    )
    parser.add_argument(
        "--scaled-candidates",
        type=int,
        default=200,
        help="Target candidate count for scaled_cache workload",
    )
    parser.add_argument(
        "--scaled-tasks",
        type=int,
        default=3,
        help="Number of sequential reuse tasks for scaled_cache workload",
    )
    parser.add_argument(
        "--scaled-coeff-variants",
        type=int,
        default=None,
        help="Override coefficient variants for scaled_cache workload (auto if omitted)",
    )
    parser.add_argument(
        "--scaled-validation-shots",
        type=int,
        default=None,
        help="Validation shots for scaled_cache workload (default: elimination_shots/4)",
    )
    args, extra = parser.parse_known_args()

    install_deps()
    load_modules()
    set_deterministic_seed(args.seed)

    # Determine output directory: use provided path or create timestamped folder
    if args.output_dir is None:
        quantum_eye_test = PERSISTENT / "Quantum-eye-test"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = quantum_eye_test / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[run] mode={args.mode}, runs={args.runs}, seed={args.seed}, workload={args.workload}")
    log(f"[output] writing results to {args.output_dir}")

    adapter_info: Dict[str, Any] | None = None
    if args.mode == "adapter":
        adapter_info = maybe_install_adapter(args.adapter_path)
        log(f"[adapter] status={adapter_info}")

    runs: List[Dict[str, Any]] = []
    if args.workload == "energy":
        for idx in range(args.runs):
            run_seed = args.seed + idx
            log(f"[run {idx}] seed={run_seed}")
            runs.append(simulate_run(run_seed))
        summary = aggregate_runs(runs)
    elif args.workload == "vqe":
        from vqe_workload import run_vqe_lih

        for idx in range(args.runs):
            run_seed = args.seed + idx
            log(f"[run {idx}] seed={run_seed} (vqe) mode={args.mode}")
            res = run_vqe_lih(
                mode=args.mode,
                seed=run_seed,
                shots=args.shots_per_eval,
                max_iter=args.vqe_iterations,
                spsa_a0=args.spsa_a0,
                spsa_c0=args.spsa_c0,
                qe_smoothing_window=args.qe_smoothing_window,
                qe_uniform_floor=args.qe_uniform_floor,
                estimator=args.vqe_estimator,
                noise_type=args.noise_type,
                noise_level=args.noise_level,
            )

            energy_trace = res.extra.get("energy_trace", []) if isinstance(res.extra, dict) else []
            best_energy = min(energy_trace) if energy_trace else res.energy
            best_iter = energy_trace.index(best_energy) if energy_trace else res.iterations
            best_energy_error = best_energy - res.reference_energy

            runs.append(
                {
                    "metrics": {
                        "energy": res.energy,
                        "reference_energy": res.reference_energy,
                        "energy_error": res.energy_error,
                        "best_energy": best_energy,
                        "best_energy_error": best_energy_error,
                        "best_iteration": best_iter,
                        "wall_clock_s": res.wall_clock_s,
                        "shots_used": res.shots,
                        "total_shots": res.shots,
                        "iterations": res.iterations,
                        "fidelity_to_reference": res.fidelity_to_reference,
                        "max_rss_mb": res.max_rss_mb,
                        "success": res.success,
                    },
                    "config": res.extra,
                }
            )

        summary = aggregate_runs(runs)
    elif args.workload == "bell":
        for idx in range(args.runs):
            run_seed = args.seed + idx
            log(f"[run {idx}] seed={run_seed} (bell)")
            runs.append(run_bell(mode=args.mode, seed=run_seed))
        summary = aggregate_runs(runs)
    elif args.workload == "hamiltonian_elimination":
        from hamiltonian_elimination import run_elimination_experiment

        elimination = run_elimination_experiment(
            n_qubits=args.elimination_qubits,
            seed=args.seed,
            shots=args.elimination_shots,
            validation_probes=None,
            validation_shots=max(256, args.elimination_shots // 4),
            coeff_variants=args.elimination_coeff_variants,
        )
        cache = elimination["cache"]
        cache_json = args.output_dir / "constraint_cache.json"
        cache.to_json(cache_json)
        elim_json = args.output_dir / "elimination_results.json"
        elim_payload = {
            "experiment": elimination["experiment"],
            "cache": cache.to_dict(),
        }
        elim_json.write_text(json.dumps(elim_payload, indent=2))
        log(f"[elimination] wrote cache to {cache_json}")
        runs = [elimination["experiment"]]
        summary = {
            "initial_candidates": elimination["experiment"]["initial_count"],
            "final_candidates": elimination["experiment"]["final_count"],
        }
    elif args.workload == "scaled_cache":
        from scaled_experiment import run_scaled_experiment_sweep

        scaled = run_scaled_experiment_sweep(
            base_seed=args.seed,
            seeds=args.scaled_seeds,
            candidate_target=args.scaled_candidates,
            tasks=args.scaled_tasks,
            n_qubits=args.elimination_qubits,
            coeff_variants=args.scaled_coeff_variants or args.elimination_coeff_variants,
            shots=args.elimination_shots,
            validation_shots=args.scaled_validation_shots
            if args.scaled_validation_shots is not None
            else max(256, args.elimination_shots // 4),
            output_dir=args.output_dir,
        )
        runs = []
        # Persist CSV requires metrics/config keys; fill with placeholders plus reuse stats.
        csv_fields = {
            "energy": None,
            "reference_energy": None,
            "energy_error": None,
            "best_energy": None,
            "best_energy_error": None,
            "best_iteration": None,
            "fidelity_to_reference": None,
            "wall_clock_s": None,
            "shots_used": None,
            "total_shots": None,
            "max_rss_mb": None,
            "iterations": None,
            "x_accuracy": None,
            "y_accuracy": None,
            "x_correlation": None,
            "y_correlation": None,
            "qsv_score": None,
            "qsv_P": None,
            "qsv_S": None,
            "qsv_E": None,
            "qsv_Q": None,
            "success": True,
        }
        for seed_run in scaled.get("per_seed", []):
            reuse_summary = seed_run.get("reuse", {}).get("summary", {}) if isinstance(seed_run, dict) else {}
            shots_used = reuse_summary.get("total_cache_shots")
            total_shots = reuse_summary.get("total_cache_shots")
            metrics = dict(csv_fields)
            metrics["shots_used"] = shots_used
            metrics["total_shots"] = total_shots
            metrics["wall_clock_s"] = reuse_summary.get("total_baseline_shots")
            metrics["success"] = True
            runs.append(
                {
                    "config": {"seed": seed_run.get("seed")},
                    "metrics": metrics,
                    "extra": seed_run,
                }
            )
        summary = scaled.get("summary", {})
    else:  # cache_reuse
        from cache_reuse_demo import run_cache_reuse_demo

        cache_path = args.cache_path
        if cache_path is None:
            cache_path = args.output_dir / "constraint_cache.json"
        reuse_res = run_cache_reuse_demo(
            cache_path=cache_path,
            seed=args.seed,
            n_qubits=args.reuse_qubits,
        )
        reuse_json = args.output_dir / "cache_reuse_results.json"
        reuse_json.write_text(json.dumps(reuse_res, indent=2))
        log(f"[reuse] loaded cache from {cache_path}")
        runs = [reuse_res]
        summary = reuse_res.get("summary", {})

    pip_state = get_pip_freeze()

    metadata = {
        "mode": args.mode,
        "base_seed": args.seed,
        "runs": args.runs,
        "ffsim_version": getattr(ffsim, "__version__", "unknown"),
        "ffsim_git_sha": get_git_sha(FFSIM_SRC),
        "ffsim_source": "pip (pinned)",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "base_image": get_base_image(),
        "image_digest": get_image_digest(),
        "pip_freeze": pip_state,
        "hardware": get_hardware_info(),
        "env_sample": {k: os.environ.get(k) for k in ["JUPYTER_PORT", "NB_USER", "HOSTNAME"]},
        "adapter": adapter_info,
        "pip_extra": os.environ.get("FFSIM_PIP_EXTRA", ""),
        "extra_args": extra,
    }

    results = {
        "mode": args.mode,
        "metadata": metadata,
        "runs": runs,
        "summary": summary,
    }

    json_path = args.output_dir / f"results-{args.mode}.json"
    csv_path = args.output_dir / f"results-{args.mode}.csv"
    json_path.write_text(json.dumps(results, indent=2))
    persist_csv(runs, csv_path)

    log(f"[done] wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()

