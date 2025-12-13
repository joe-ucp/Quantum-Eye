from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from constraint_cache import ConstraintCache
from hamiltonian_elimination import (
    CandidateHamiltonian,
    expectation_from_state,
    generate_candidate_family,
    ground_state,
)


def _evaluate_candidates(candidates: List[CandidateHamiltonian], n_qubits: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    evaluations: List[Dict[str, Any]] = []
    for cand in candidates:
        energy, _ = ground_state(cand.terms, n_qubits)
        evaluations.append(
            {
                "structure_id": cand.structure_id,
                "coeff_id": cand.coeff_id,
                "energy": energy,
                "terms": cand.term_labels,
            }
        )
    best = min(evaluations, key=lambda x: x["energy"]) if evaluations else None
    return evaluations, best


def _measurement_cost_params(cache: ConstraintCache) -> Tuple[int, float]:
    """Return (observables_per_candidate, shots_per_observable) from cached bounds."""
    obs_per_candidate = max(1, len(cache.measurement_bounds))
    if cache.measurement_bounds:
        shots_per_obs = cache.metadata.get("shots", 1) / max(1, len(cache.measurement_bounds))
    else:
        shots_per_obs = 1
    return obs_per_candidate, shots_per_obs


def _filter_candidates_with_cache(
    candidates: List[CandidateHamiltonian], cache: ConstraintCache, n_qubits: int
) -> List[CandidateHamiltonian]:
    filtered: List[CandidateHamiltonian] = []
    observables = list(cache.measurement_bounds.keys())
    for cand in candidates:
        expectations = None
        if observables:
            _, state = ground_state(cand.terms, n_qubits)
            expectations = {obs: expectation_from_state(obs, state) for obs in observables}
        if cache.is_candidate_allowed(
            cand.term_labels,
            candidate_symmetry=cand.symmetry,
            candidate_reachability=cand.reachability,
            expectations=expectations,
        ):
            filtered.append(cand)
    return filtered


def run_cache_reuse_demo(
    cache_path: Path, seed: int = 12345, n_qubits: int | None = None
) -> Dict[str, Any]:
    cache = ConstraintCache.from_json(Path(cache_path))
    n_q = int(cache.metadata.get("n_qubits", 6)) if n_qubits is None else n_qubits

    # Fresh candidate family (different seed to represent a new task).
    family = generate_candidate_family(n_qubits=n_q, seed=seed + 999)

    constrained_candidates = _filter_candidates_with_cache(family, cache, n_q)
    unconstrained_eval, unconstrained_best = _evaluate_candidates(family, n_q)
    constrained_eval, constrained_best = _evaluate_candidates(constrained_candidates, n_q)

    obs_per_candidate, shots_per_obs = _measurement_cost_params(cache)
    baseline_measurements = len(family) * obs_per_candidate
    constrained_measurements = len(constrained_candidates) * obs_per_candidate
    shots_saved = (baseline_measurements - constrained_measurements) * shots_per_obs

    summary = {
        "cache_path": str(cache_path),
        "n_qubits": n_q,
        "unconstrained_candidates": len(family),
        "constrained_candidates": len(constrained_candidates),
        "pruned": len(family) - len(constrained_candidates),
        "prune_ratio": 0.0 if len(family) == 0 else (len(family) - len(constrained_candidates)) / len(family),
        "best_energy_unconstrained": None if unconstrained_best is None else unconstrained_best["energy"],
        "best_energy_constrained": None if constrained_best is None else constrained_best["energy"],
        "baseline_measurements": baseline_measurements,
        "cache_measurements": constrained_measurements,
        "shots_saved_estimate": shots_saved,
    }

    return {
        "summary": summary,
        "unconstrained": unconstrained_eval,
        "constrained": constrained_eval,
        "best_unconstrained": unconstrained_best,
        "best_constrained": constrained_best,
    }


def run_multi_task_reuse(
    cache_path: Path,
    tasks: int = 3,
    base_seed: int = 12345,
    n_qubits: int | None = None,
    coeff_variants: int = 3,
) -> Dict[str, Any]:
    """
    Run sequential cache reuse across multiple tasks to demonstrate cumulative benefit.
    """
    cache = ConstraintCache.from_json(Path(cache_path))
    n_q = int(cache.metadata.get("n_qubits", 6)) if n_qubits is None else n_qubits
    obs_per_candidate, shots_per_obs = _measurement_cost_params(cache)

    total_measurements_baseline = 0
    total_measurements_cache = 0
    total_shots_baseline = 0.0
    total_shots_cache = 0.0
    task_summaries: List[Dict[str, Any]] = []

    for idx in range(tasks):
        task_seed = base_seed + idx * 1000 + 999  # offset to differ from elimination seed
        family = generate_candidate_family(n_qubits=n_q, seed=task_seed, coeff_variants=coeff_variants)
        constrained_candidates = _filter_candidates_with_cache(family, cache, n_q)

        unconstrained_eval, unconstrained_best = _evaluate_candidates(family, n_q)
        constrained_eval, constrained_best = _evaluate_candidates(constrained_candidates, n_q)

        baseline_measurements = len(family) * obs_per_candidate
        cache_measurements = len(constrained_candidates) * obs_per_candidate
        shots_saved = (baseline_measurements - cache_measurements) * shots_per_obs
        baseline_shots = baseline_measurements * shots_per_obs
        cache_shots = cache_measurements * shots_per_obs

        total_measurements_baseline += baseline_measurements
        total_measurements_cache += cache_measurements
        total_shots_baseline += baseline_shots
        total_shots_cache += cache_shots

        task_summaries.append(
            {
                "task_index": idx,
                "seed": task_seed,
                "unconstrained_candidates": len(family),
                "constrained_candidates": len(constrained_candidates),
                "pruned": len(family) - len(constrained_candidates),
                "prune_ratio": 0.0
                if len(family) == 0
                else (len(family) - len(constrained_candidates)) / len(family),
                "baseline_measurements": baseline_measurements,
                "cache_measurements": cache_measurements,
                "baseline_shots": baseline_shots,
                "cache_shots": cache_shots,
                "shots_saved_estimate": shots_saved,
                "unconstrained": unconstrained_eval,
                "constrained": constrained_eval,
                "best_unconstrained": unconstrained_best,
                "best_constrained": constrained_best,
                "best_energy_unconstrained": None if unconstrained_best is None else unconstrained_best["energy"],
                "best_energy_constrained": None if constrained_best is None else constrained_best["energy"],
            }
        )

    total_shots_saved = total_shots_baseline - total_shots_cache
    summary = {
        "cache_path": str(cache_path),
        "n_qubits": n_q,
        "tasks": tasks,
        "obs_per_candidate": obs_per_candidate,
        "shots_per_observable": shots_per_obs,
        "total_baseline_measurements": total_measurements_baseline,
        "total_cache_measurements": total_measurements_cache,
        "total_baseline_shots": total_shots_baseline,
        "total_cache_shots": total_shots_cache,
        "total_shots_saved_estimate": total_shots_saved,
        "unconstrained_candidates_per_task": [t["unconstrained_candidates"] for t in task_summaries],
        "constrained_candidates_per_task": [t["constrained_candidates"] for t in task_summaries],
        "prune_ratios": [t["prune_ratio"] for t in task_summaries],
    }

    return {
        "summary": summary,
        "tasks": task_summaries,
    }


if __name__ == "__main__":  # pragma: no cover - manual check
    demo_cache = Path("constraint_cache.example.json")
    if not demo_cache.exists():
        raise SystemExit("No cache file found; run hamiltonian_elimination first.")
    result = run_cache_reuse_demo(cache_path=demo_cache)
    print(result["summary"])

