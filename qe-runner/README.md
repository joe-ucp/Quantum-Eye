# Persistent Volume Structure

This directory contains experiment scripts and data for ffsim + Quantum Eye adapter testing.

## Directory Structure

- **Scripts** (root level):
  - `run_experiment.py` - Main experiment runner
  - `compare_results.py` - Compare baseline vs adapter results
  - `vqe_workload.py` - VQE workload implementation
  - `README_NOISE.md` - Noise model documentation

- **Quantum-eye-test/** - New experiment runs
  - Each run creates a timestamped subfolder: `YYYYMMDD_HHMMSS/`
  - Contains: `results-baseline.json/csv`, `results-adapter.json/csv`, `comparison_summary.txt/json`, `comparison_plot.png`

- **archive/** - Archived experiment data
  - Contains all historical results and comparison files from previous runs

- **adapter/** - Adapter code mounted for editable installs

## Usage

### Running Experiments

New experiments automatically create timestamped folders under `Quantum-eye-test/`:

```bash
# Baseline run (creates Quantum-eye-test/YYYYMMDD_HHMMSS/results-baseline.json)
docker compose -f compose.yaml -f compose.test.yaml run --rm notebook \
  --workload vqe --mode baseline

# Adapter run (creates Quantum-eye-test/YYYYMMDD_HHMMSS/results-adapter.json)
docker compose -f compose.yaml -f compose.adapter.yaml run --rm notebook \
  --workload vqe --mode adapter
```

### Comparing Results

The compare script automatically finds the most recent baseline/adapter pair in `Quantum-eye-test/`:

```bash
docker compose -f compose.yaml -f compose.adapter.yaml run --rm --entrypoint python notebook \
  /home/jovyan/persistent-volume/compare_results.py
```

Or specify explicit paths:

```bash
docker compose -f compose.yaml -f compose.adapter.yaml run --rm --entrypoint python notebook \
  /home/jovyan/persistent-volume/compare_results.py \
  --baseline /home/jovyan/persistent-volume/Quantum-eye-test/20240101_120000/results-baseline.json \
  --adapter /home/jovyan/persistent-volume/Quantum-eye-test/20240101_120000/results-adapter.json
```

### Overriding Output Directory

To write outputs to a specific location:

```bash
docker compose -f compose.yaml -f compose.test.yaml run --rm notebook \
  --workload vqe --mode baseline \
  --output-dir /home/jovyan/persistent-volume/custom-folder
```

