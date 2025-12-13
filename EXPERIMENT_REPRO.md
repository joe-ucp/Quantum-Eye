# Scaled Constraint-Cache Repro (Docker)

Point-of-entry to rerun the scaled cache experiment (200 candidates → ≤3 survivors) using Docker compose (Python 3.11 image). Custom code lives in `qe-runner/`, separate from the upstream `ffsim` tree.

## Prereqs
- Docker + docker compose
- Working dir: repo root (`Quantum-Eye/`)
- Clone ffsim alongside: `git clone https://github.com/qiskit-community/ffsim.git ffsim`

## Command (single-seed canonical run)
Run from repo root:
```
docker compose -f ffsim/compose.yaml -f compose.qe.yaml run --rm notebook \
  --mode baseline --workload scaled_cache \
  --seed 12345 --scaled-seeds 1 --scaled-candidates 200 --scaled-tasks 3 \
  --elimination-coeff-variants 50 --elimination-shots 8192 \
  --output-dir /home/jovyan/persistent-volume/artifacts/scaled_single_seed_shots8192
```

What this does:
- Mounts `qe-runner/` into the container and uses `/home/jovyan/qe-runner/run_experiment.py` as entrypoint.
- Mounts `persistent-volume/` for outputs and adapter data.

## Outputs
- JSON/CSV under `ffsim/persistent-volume/artifacts/scaled_single_seed_shots8192/`
  - `results-baseline.json` contains `summary` (aggregate) and per-seed details.
  - `results-baseline.csv` is a slim table for quick inspection.
- Large artifacts are gitignored; keep them untracked.

## What to read in the summary
- `candidate_target` / `coeff_variants_used`: intended family size (200 via 4 structures × 50 variants).
- `mean_final_count`: survivors after elimination + validation (goal ≤3).
- `mean_prune_ratio`: fraction removed (goal ≥0.98 for 200 → ≤4).
- `mean_shots_saved_ratio`: 1 − cache_shots/baseline_shots (goal ≥0.8).
- `gate_checks`: boolean pass/fail for repeatability, non_triviality, honest_baseline.

## Reference result (latest canonical run)
From `results-baseline.json` produced with the command above:
- `mean_final_count`: **2**
- `mean_prune_ratio`: **0.99**
- `mean_shots_saved_ratio`: **0.9967**
- `total_baseline_shots`: **4,915,200**
- `total_cache_shots`: **16,384**
- `gate_checks`: `{repeatability: true, non_triviality: true, honest_baseline: true}`

Interpretation: baseline scales with 200 candidates (~600 probe circuits × 8192 shots); cache scales with 2 survivors (~2 probe circuits × 8192 shots), yielding ~300× shot reduction.

