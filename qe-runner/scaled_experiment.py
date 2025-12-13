from __future__ import annotations

import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cache_reuse_demo import run_multi_task_reuse
from hamiltonian_elimination import _base_structures, run_elimination_experiment


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _choose_coeff_variants(n_qubits: int, target_candidates: int, override: int | None) -> int:
    if override is not None and override > 0:
        return override
    structures = _base_structures(n_qubits)
    num_structures = max(1, len(structures))
    variants = math.ceil(target_candidates / num_structures)
    return max(variants, 3)


def run_scaled_experiment_sweep(
    *,
    base_seed: int = 12345,
    seeds: int = 5,
    candidate_target: int = 200,
    tasks: int = 3,
    n_qubits: int = 6,
    coeff_variants: int | None = None,
    shots: int = 4096,
    validation_shots: int | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run scaled elimination + multi-task cache reuse across multiple seeds.

    Returns a dict containing per-seed results and aggregate gate checks.
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    variants = _choose_coeff_variants(n_qubits, candidate_target, coeff_variants)
    vshots = validation_shots if validation_shots is not None else max(256, shots // 4)

    per_seed: List[Dict[str, Any]] = []
    final_counts: List[float] = []
    prune_ratios: List[float] = []
    shots_saved_ratios: List[float] = []

    total_baseline_shots = 0.0
    total_cache_shots = 0.0

    for idx in range(seeds):
        seed = base_seed + idx
        elim = run_elimination_experiment(
            n_qubits=n_qubits,
            seed=seed,
            shots=shots,
            validation_probes=None,
            validation_shots=vshots,
            coeff_variants=variants,
        )
        cache = elim["cache"]
        experiment = elim["experiment"]

        cache_path: Path | None = None
        if output_dir is not None:
            cache_path = output_dir / f"constraint_cache_seed{seed}.json"
            cache.to_json(cache_path)
            elim_path = output_dir / f"elimination_results_seed{seed}.json"
            elim_payload = {"experiment": experiment, "cache": cache.to_dict()}
            elim_path.write_text(json.dumps(elim_payload, indent=2))

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_cache_path = Path(tmpdir) / "cache.json"
                cache.to_json(temp_cache_path)
                reuse = run_multi_task_reuse(
                    cache_path=temp_cache_path,
                    tasks=tasks,
                    base_seed=base_seed + idx * 10000,
                    n_qubits=n_qubits,
                    coeff_variants=variants,
                )
        else:
            reuse = run_multi_task_reuse(
                cache_path=cache_path,
                tasks=tasks,
                base_seed=base_seed + idx * 10000,
                n_qubits=n_qubits,
                coeff_variants=variants,
            )

        if output_dir is not None:
            reuse_path = output_dir / f"cache_reuse_results_seed{seed}.json"
            reuse_path.write_text(json.dumps(reuse, indent=2))

        initial_count = experiment["initial_count"]
        final_count = experiment["final_count"]
        prune_ratio = 0.0 if initial_count == 0 else (initial_count - final_count) / initial_count
        final_counts.append(float(final_count))
        prune_ratios.append(float(prune_ratio))

        baseline_shots = reuse["summary"]["total_baseline_shots"]
        cache_shots = reuse["summary"]["total_cache_shots"]
        shots_saved = reuse["summary"]["total_shots_saved_estimate"]
        shots_saved_ratio = 0.0 if baseline_shots == 0 else shots_saved / baseline_shots
        shots_saved_ratios.append(float(shots_saved_ratio))

        total_baseline_shots += baseline_shots
        total_cache_shots += cache_shots

        per_seed.append(
            {
                "seed": seed,
                "coeff_variants": variants,
                "elimination": experiment,
                "cache": cache.to_dict(),
                "reuse": reuse,
                "prune_ratio": prune_ratio,
                "shots_saved_ratio": shots_saved_ratio,
            }
        )

    mean_final, std_final = _mean_std(final_counts)
    mean_prune, std_prune = _mean_std(prune_ratios)
    mean_saved, std_saved = _mean_std(shots_saved_ratios)

    total_shots_saved = total_baseline_shots - total_cache_shots
    total_saved_ratio = 0.0 if total_baseline_shots == 0 else total_shots_saved / total_baseline_shots

    gate_checks = {
        "repeatability": std_final <= 1.0 and std_prune <= 0.1,
        "non_triviality": (min(final_counts) if final_counts else 0) <= 3
        and (min(prune_ratios) if prune_ratios else 0) >= 0
        and (min(per_seed, key=lambda p: p["elimination"]["initial_count"])["elimination"]["initial_count"]
             if per_seed else 0)
        >= candidate_target,
        "honest_baseline": total_saved_ratio >= 0.8,
    }

    summary = {
        "base_seed": base_seed,
        "seeds": seeds,
        "n_qubits": n_qubits,
        "candidate_target": candidate_target,
        "coeff_variants_used": variants,
        "tasks": tasks,
        "shots": shots,
        "validation_shots": vshots,
        "mean_final_count": mean_final,
        "std_final_count": std_final,
        "mean_prune_ratio": mean_prune,
        "std_prune_ratio": std_prune,
        "mean_shots_saved_ratio": mean_saved,
        "std_shots_saved_ratio": std_saved,
        "total_baseline_shots": total_baseline_shots,
        "total_cache_shots": total_cache_shots,
        "total_shots_saved_estimate": total_shots_saved,
        "total_shots_saved_ratio": total_saved_ratio,
        "gate_checks": gate_checks,
    }

    return {
        "summary": summary,
        "per_seed": per_seed,
    }

