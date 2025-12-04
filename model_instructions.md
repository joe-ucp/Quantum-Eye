# Model Instructions: Quantum Eye H₂ VQE Iterative Experimentation

## Purpose

This document defines the protocol for Codex to iteratively run H₂ ground-state VQE experiments under UCP/tautology constraints, maintaining a chained `results.md` log that links each run to prior results and theory.

---

## Origin Parable

On a lab bench sit two spectrometers wired to the same hydrogen sample. One uses a single narrow filter; the other requires three different filter wheels and many exposures. The apprentice can only touch the dials on the front panel, not the wiring inside. Readouts spill onto loose slips: raw voltages, angles, timestamps, plots, and one JSON ledger with the last calibrated energy estimates for H₂. But there is no bound notebook; no one checks prior slips before another run, and the four UCP gauges—L, C, I, U—are marked on the wall yet rarely consulted. The lab drifts, coherent but forgetful.

## Target Parable

The same bench now has a single bound logbook beside the spectrometers. Before touching any dial, the apprentice reads the last page, compares today's target to yesterday's H₂ energy traces, and notes which UCP gauge was weakest. They still only adjust front-panel settings—exposure time, filter angle, shot count—but after each run they paste in the new plots, record the JSON summary, and write a short reflection on how contradictions shrank: the single-filter device matching the three-filter rig while staying faithful to the theory manual on the shelf. Every page references the previous one; the logbook itself is the instrument's memory and proof.

---

## Context

- **Repo**: `joe-ucp/quantum-eye` (assume local clone already present)
- **Theory**: Papers in `papers/` directory:
  - `Tautology Maximization Toward Quantum Style.pdf`
  - `UCP Cheat Sheet.pdf`
  - `Quantum Eye Complete Quantum State Recovery from Single-Basis Measurements via Frequency Signatures.pdf`
- **Test**: `tests/test_VQE_Multibasis.py` defines `H2VQESingleBasisValidationTest`
- **Current behavior**: One Z-basis VQE run plus traditional XX/YY runs; outputs figures and `h2_vqe_single_basis_results.json`
- **Target exact energy**: ≈ -1.137 Ha (ground state of H₂ at bond length 0.735 Å)
- **Current single-basis energy**: ≈ -1.043 Ha (from latest run)

**Note on UCP Components**: The codebase uses P, S, E, Q (Phase Coherence, State Distribution, Entropic Measures, Quantum Correlations), while the parables reference L, C, I, U. Treat these as isomorphic: P↔L (Logical Consistency), S↔C (Computational Closure), E↔I (Information Conservation), Q↔U (Unity). The QSV condition is P×S×E×Q>0.

---

## Contradictions to Resolve

1. **Editing-anywhere vs tests-only scope** → Resolve by strict test-file/variable editing rules
2. **One-off runs vs history-aware reasoning** → Resolve by always reading prior JSON/Markdown before each new test
3. **Pure energy minimization vs tautology-maximizing UCP loop** → Resolve by defining contradiction index and requiring non-increasing contradiction across runs
4. **Numeric artifacts only vs human-auditable reasoning** → Resolve by generating `results.md` that ties numbers to UCP and Quantum Eye theory
5. **Quantum Eye as black box vs papers/ as ground truth** → Resolve by treating papers/ as read-only spec

---

## Protocol: 10-Step Iteration Cycle

### Step 1: Initialize Environment and Repo

**Action**: Ensure Python environment is active; install dependencies from `requirements.txt`; confirm tests can be discovered.

**Commands**:
```bash
# Activate virtual environment if needed
# source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify test discovery (do not run yet)
python -m unittest discover -s tests -p "test_VQE*.py" --dry-run
```

**Validation**: No import errors; test file is discoverable.

**Resolves**: spec

---

### Step 2: Fix Editing Scope Ex Ante

**Action**: Establish immutable boundaries.

**Read-only zones** (NEVER edit):
- `adapters/` (including `quantum_eye_adapter.py`)
- `core/` (all Quantum Eye core modules)
- `papers/` (all PDFs and theory documents)
- `quantum_eye.py` (main framework entry point)
- `utils/` (utility modules)
- Any non-test Python files

**Editable zone** (ONLY edit):
- `tests/test_VQE_Multibasis.py` (or any test file in `tests/`)

**Allowed edits within tests**:
- Numeric variables: `optimal_theta`, `default_shots`, `noise_level`
- Configuration dictionaries: `config` in `setUpClass` (backend_name, shots, noise_level)
- Test method parameters (if any)
- **DO NOT**: Change function signatures, control flow, QuantumEyeAdapter internals, Hamiltonian coefficients, exact energy reference

**Resolves**: scope

---

### Step 3: Load Theoretical Spec

**Action**: Read theory documents to refresh UCP definitions, contradiction index, P/S/E/Q, QSV condition.

**Files to reference** (read-only, do not edit):
- `papers/Tautology Maximization Toward Quantum Style.pdf`
- `papers/UCP Cheat Sheet.pdf`
- `papers/Quantum Eye Complete Quantum State Recovery from Single-Basis Measurements via Frequency Signatures.pdf`

**Key concepts to extract**:
- **UCP Components**: P (Phase Coherence), S (State Distribution), E (Entropic Measures), Q (Quantum Correlations)
- **QSV Condition**: P×S×E×Q > 0 (physical validity constraint)
- **Tautology Maximization**: Parser → Critic → Repair → Validator loop
- **Contradiction Index (CI)**: Measure of deviation from exact energy and cross-basis consistency

**Resolves**: spec

---

### Step 4: Establish Current Baseline from Latest Run

**Action**: Locate the most recent `h2_vqe_validation_results_*` directory (by timestamp); read JSON and any existing `results.md`.

**Commands**:
```bash
# Find latest results directory
ls -td h2_vqe_validation_results_* | head -1
# Or on Windows:
# Get-ChildItem -Directory -Filter "h2_vqe_validation_results_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

**Extract from JSON**:
- `energies.single_basis`, `energies.traditional`, `energies.ideal`, `energies.exact`
- `errors.single_basis_error`, `errors.traditional_error`
- `expectation_values.single_basis`, `expectation_values.traditional`, `expectation_values.ideal`
- `optimal_theta`, `backend`, `resource_usage`

**Extract from `results.md`** (if exists):
- Previous CI value (retrieve using: `grep -oP "(?<=<!-- CI: ).*" $(ls -td h2_vqe_validation_results_*/results.md | head -1)` on Linux/Mac, or parse manually on Windows)
- Previous reasoning about UCP components
- Previous parameter settings that worked best

**If no `results.md` exists**: Synthesize baseline from JSON only; note this is the first log entry.

**Resolves**: history

---

### Step 5: Define Contradiction Index for H₂

**Action**: Compute a scalar contradiction index from the current JSON.

**Formula**:
```
CI = |E_single - E_exact| + α * |E_single - E_traditional| + β * cross_basis_mismatch
```

Where:
- `E_single` = `energies.single_basis`
- `E_exact` = `energies.exact` (≈ -1.137 Ha)
- `E_traditional` = `energies.traditional`
- `cross_basis_mismatch` = |exp_XX_single - exp_XX_traditional| + |exp_YY_single - exp_YY_traditional|
- `α = 0.1` (weight for single-vs-traditional discrepancy)
- `β = 0.05` (weight for cross-basis prediction error)

**Additional constraints**:
- If QSV condition fails (P×S×E×Q ≤ 0), add penalty: `CI += 1.0`
- If `single_basis_error > 0.15 Ha`, mark as high contradiction

**Tautology Energy**: `ET = 1 - CI` (for reasoning; not stored, but used to judge improvement)

**Resolves**: tautology

---

### Step 6: Choose Next Experiment by Adjusting Only Test Variables

**Action**: Based on baseline CI and prior `results.md`, select minimal change in allowed test parameters.

**Decision tree**:

1. **If CI decreased in last run**: Continue in same direction (e.g., if `optimal_theta` improved, refine it further)
2. **If CI increased**: Revert to best-known settings from prior `results.md`; try alternative parameter
3. **If first run**: Start with default `optimal_theta = 0.9 * π` (≈ 2.827)

**Allowed parameter adjustments**:
- `optimal_theta`: Adjust by ±0.1π increments (e.g., 0.8π, 0.9π, 1.0π, 1.1π)
- `default_shots`: Try 4096, 8192, 16384, 32768
- `noise_level`: Try 0.0, 0.01, 0.02 (if testing noise robustness)
- `backend_name`: Try `'aer_simulator'` (default) or other simulators if available

**UCP reasoning** (internal, document in `results.md`):
- More shots → higher I (Information Conservation)
- Lower noise → higher L (Logical Consistency)
- Better `optimal_theta` → higher U (Unity between single-basis and traditional)
- Cross-basis prediction accuracy → higher C (Computational Closure)

**Resolves**: scope

---

### Step 7: Run the H₂ VQE Validation Test

**Action**: Execute the test after applying chosen parameter changes.

**Command**:
```bash
python -m unittest tests.test_VQE_Multibasis.H2VQESingleBasisValidationTest.test_h2_vqe_single_basis_energy_calculation -v
```

**Expected outputs**:
- New timestamped directory: `h2_vqe_validation_results_YYYYMMDD_HHMMSS/`
- `h2_vqe_single_basis_results.json`
- `h2_vqe_single_basis_validation.png`
- `h2_cross_basis_prediction.png`

**Do not modify production code** even if results are surprising; prefer changing test parameters on subsequent iterations.

**Resolves**: tautology

---

### Step 8: Generate or Extend `results.md` in New Results Folder

**Action**: Create `results.md` in the just-created `h2_vqe_validation_results_*` directory.

**Template**:

```markdown
# H₂ VQE Single-Basis Validation Results

**Run Date**: YYYY-MM-DD HH:MM:SS  
**Results Directory**: `h2_vqe_validation_results_YYYYMMDD_HHMMSS/`

## Previous Run Summary

[If not first run, summarize:]
- **Previous CI**: [value]
- **Previous single-basis energy**: [value] Ha
- **Previous error**: [value] Ha
- **Previous best parameters**: `optimal_theta = [value]`, `shots = [value]`

## Current Run Parameters

- `optimal_theta`: [value] (≈ [value]π)
- `default_shots`: [value]
- `noise_level`: [value]
- `backend`: [value]

## Current Run Results

### Energies
- **Single-basis**: [value] Ha
- **Traditional**: [value] Ha
- **Ideal**: [value] Ha
- **Exact**: [value] Ha

### Errors
- **Single-basis error**: [value] Ha
- **Traditional error**: [value] Ha

### Expectation Values
- **Single-basis**: II=[value], ZI=[value], IZ=[value], ZZ=[value], XX=[value], YY=[value]
- **Traditional**: II=[value], ZI=[value], IZ=[value], ZZ=[value], XX=[value], YY=[value]
- **Ideal**: II=[value], ZI=[value], IZ=[value], ZZ=[value], XX=[value], YY=[value]

### Contradiction Index (CI)
- **Current CI**: [value]
- **Previous CI**: [value] (if applicable)
- **Change**: [decreased/increased/unchanged] by [delta]

### UCP Component Analysis

**Phase Coherence (P) / Logical Consistency (L)**:
- [Qualitative assessment: How well do single-basis predictions match traditional measurements?]
- [Quantitative: Cross-basis mismatch = |XX_single - XX_trad| + |YY_single - YY_trad| = [value]]

**State Distribution (S) / Computational Closure (C)**:
- [Assessment: How efficiently does single-basis method achieve same result as 3-circuit traditional?]
- [Resource savings: 66.7% (1 circuit vs 3 circuits)]

**Entropic Measures (E) / Information Conservation (I)**:
- [Assessment: How much information is preserved from Z-basis measurement?]
- [Shot count: [value] shots]

**Quantum Correlations (Q) / Unity (U)**:
- [Assessment: How well does the method maintain consistency with exact ground state?]
- [Energy error: [value] Ha]

**QSV Condition**: P×S×E×Q > 0 [PASS/FAIL]

**Note on QSV evaluation**: If QuantumEyeAdapter doesn't yet expose P,S,E,Q metrics directly, record qualitative PASS/FAIL derived from sign of energy covariance across bases. A PASS indicates that single-basis, traditional, and ideal energies show consistent trends (positive covariance), while a FAIL indicates contradictory energy estimates that violate physical validity.

## Grounding Note

[Explicitly tie interpretations back to papers/:]
- This run [improved/degraded/maintained] contradiction reduction relative to theory expectations.
- Single-basis method [matches/diverges from] traditional multi-basis results within [tolerance].
- Energy estimate [approaches/diverges from] exact ground state (-1.137 Ha).
- Cross-basis predictions (XX, YY) [are consistent with/show deviation from] direct measurements.

**Reference to theory**: [Cite relevant section from papers/ if applicable]

## Next Steps

[If CI decreased:] Continue refining `optimal_theta` in direction of improvement.  
[If CI increased:] Revert to previous best parameters; try alternative adjustment.  
[If CI unchanged:] Explore different parameter (e.g., shot count) while maintaining current `optimal_theta`.

---

**Previous Run**: [Link to previous results.md if exists, or "First run - no previous log"]

**Note on reference handling**: If previous results.md is missing, create an empty stub in the same directory to keep link chain intact. The stub should contain at minimum: "# Previous Run (Missing)" and "This run's previous results.md was not found; treating as first run in chain."

<!-- CI: [computed_CI_value] -->
```

**Critical**: At the end of Step 8, write the computed CI value as the last line in the Markdown in machine-readable form: `<!-- CI: [value] -->`. This makes Step 4 automatically parsable for CI retrieval.

**Resolves**: md-bridge

---

### Step 9: REAL Repair Step if Contradiction Worsens

**Action**: If new CI > previous CI, explicitly mark in `results.md` that previous configuration remains best.

**Repair protocol**:
1. In `results.md`, add section:
   ```markdown
   ## Repair Action Required
   
   **Contradiction increased**: CI_new = [value] > CI_prev = [value]
   **Best-known configuration**: [restore from previous results.md]
   - `optimal_theta` = [best value]
   - `shots` = [best value]
   - `noise_level` = [best value]
   
   **Reasoning**: [Explain why contradiction increased; e.g., parameter overshoot, noise introduced, etc.]
   ```
2. Restore best-known test variable settings in `tests/test_VQE_Multibasis.py` for future runs
3. Treat this as attunement step: keep all UCP components above coherence thresholds rather than forcing progress

**Resolves**: tautology

---

### Step 10: Enforce History Linkage and Scope After Each Cycle

**Action**: Before ending execution session, verify:

1. **Scope check**:
   ```bash
   git diff --name-only
   ```
   Should show changes only in:
   - `tests/test_VQE_Multibasis.py` (or other test files)
   - New `h2_vqe_validation_results_*/` directories
   - New `results.md` files
   - **NOT** in `adapters/`, `core/`, `papers/`, `quantum_eye.py`, `utils/`

2. **History linkage check**:
   - Each new `results.md` references the immediately preceding one (or its summary)
   - Chain of reasoning is continuous (no gaps)

3. **Next invocation**: When user says "look at model_instructions.md and execute", start again at Step 1, but always read the latest `results.md` before choosing fresh test parameters.

**Resolves**: history

---

## Guardrails

### Never Modify
- Non-test code (`adapters/`, `core/`, production modules)
- Any file in `papers/` (other than adding new Markdown logs alongside, not inside, theory documents)
- Physical constants or Hamiltonian coefficients for H₂
- Reference exact energy (-1.137 Ha)

### Never Fabricate
- Do not infer or fabricate energies; always base numbers on actual test runs and persisted JSON output

### Never Overwrite
- Preserve all previous result directories and `results.md` files; never overwrite or delete prior evidence

### Optimization Constraints
- Any optimization or search over `optimal_theta` or other parameters must proceed via discrete test runs (parameter changes between runs), not by rewriting algorithmic code paths

---

## Acceptance Tests

1. **Fresh clone test**: From a fresh clone and environment, Codex can follow these instructions to run `H2VQESingleBasisValidationTest` successfully, producing a new `h2_vqe_validation_results_*` directory without touching non-test code.

2. **Results.md completeness**: Each new results directory contains plots, `h2_vqe_single_basis_results.json`, and a `results.md` that:
   - Summarizes the previous run
   - Reports new energies and expectation values
   - States whether CI and UCP balance improved, held, or worsened

3. **Energy stability**: The single-basis energy remains within a small window of the known exact value (on the order already demonstrated by existing runs) and never degrades repeatedly without being called out and repaired in `results.md`.

4. **Git diff cleanliness**: A Git diff after several cycles shows changes only in test files and new result directories/Markdown logs; `papers/` theory PDFs/MD remain unchanged.

5. **History awareness**: When prompted by the user, Codex can read `model_instructions.md`, locate the latest `results.md`, explain its last reasoning step, and propose the next minimal parameter adjustment consistent with UCP and tautology maximization.

---

## Fallbacks (If Blocked)

### Environment Setup Fails
**Action**: Switch to analysis-only mode:
1. Read the latest JSON and `results.md`
2. Compute or describe CI qualitatively
3. Append a note to the newest `results.md` explaining why no new run was performed
4. Do not edit any code

### No Prior Results Directories Exist
**Action**: Treat `papers/` theory plus the first successful H₂ test run as the initial baseline:
1. Create the first `results.md` summarizing that run
2. Explicitly state that there is no previous log page to reference
3. Use this as baseline for Step 4 in next iteration

### Test Execution Fails
**Action**: 
1. Check error message; if it's a parameter issue (e.g., invalid `optimal_theta`), adjust parameter and retry
2. If it's a code issue (e.g., import error), report to user; do not modify non-test code
3. Document failure in `results.md` if a results directory was created

---

## Notes on Protocol Steps

- **Step 1** aligns Codex with the actual project and backend so contradictions can only come from physics/results, not missing dependencies.
- **Step 2** prevents silent violations of the "tests-only" rule and keeps logical consistency (L) high by reducing edit surface.
- **Step 3** anchors Codex to formal UCP and Quantum Eye definitions so changes are checked against an external spec, not ad hoc intuition.
- **Step 4** turns scattered prior outputs into an explicit starting hypothesis, conserving information (I) and creating a history-aware loop.
- **Step 5** converts energy discrepancies and cross-basis mismatches into a contradiction index, making "ground state search" a tautology-maximizing process.
- **Step 6** confines exploration to safe parameter tweaks, increasing computational closure (C) while respecting code integrity.
- **Step 7** executes the test and generates raw data.
- **Step 8** builds the missing bridge from raw numbers to auditable reasoning, raising unity (U) across JSON, plots, and theory.
- **Step 9** enforces the REAL floor by reverting to the best-known configuration when contradictions increase, instead of overwriting good states.
- **Step 10** weaves runs into a continuous, self-consistent narrative so each invocation of Codex can shrink contradictions further while honoring all UCP components.

---

## Quick Reference: Test Variable Locations

In `tests/test_VQE_Multibasis.py`:

- **Line 42**: `'default_shots': 8192` (in `setUpClass` config)
- **Line 43**: `'noise_level': 0.0` (in `setUpClass` config)
- **Line 80**: `optimal_theta = 0.9 * np.pi` (in `test_h2_vqe_single_basis_energy_calculation`)
- **Line 91**: `shots=8192` (in `execute_circuit` call for single-basis)
- **Line 143**: `shots=8192` (in `execute_circuit` call for XX basis)
- **Line 154**: `shots=8192` (in `execute_circuit` call for YY basis)

**Hamiltonian coefficients** (lines 66-73): **DO NOT MODIFY** (fixed by chemistry)

**Exact energy** (line 76): **DO NOT MODIFY** (fixed by theory: -1.137 Ha)

---

## Execution Command

When user says "execute model_instructions.md" or "run the protocol", proceed through Steps 1-10 in sequence, starting with Step 1 (environment setup) and ending with Step 10 (verification).

**Run-loop trigger**: Codex should repeat Steps 4–10 until the CI stabilizes within 1e-3 Ha or three consecutive runs show non-decreasing CI, whichever comes first. After reaching this stopping condition, report final results and best-known configuration.

**Optional convenience**: If a Makefile is present later, a `make run-h2-vqe` command could wrap Steps 6–8 for quick iteration.

---

**End of Model Instructions**

