# Quantum Eye — Bell-State Observable Reproduction (10-Minute Experiment)

This document presents a **single-notebook, fully reproducible experiment** for estimating a Bell-state observable on IBM hardware (or a simulator), with **rigorous Hoeffding error bars**.  
Execution is entirely from:

- **Notebook:** `Notebooks/multi-basis_bell.ipynb`  
- **Repo:** [https://github.com/joe-ucp/Quantum-Eye](https://github.com/joe-ucp/Quantum-Eye)

**Objectives:**  
- Clarity  
- Reproducibility  
- Single environment, single notebook, "Run All"

---

## 1. Reproduce in ~10 Minutes

A reviewer can re-run the experiment with these steps.

### 1.1 Prerequisites (2–3 minutes)
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime numpy matplotlib pandas seaborn
```

### 1.2 Clone and Open the Notebook (1–2 minutes)
```bash
git clone https://github.com/joe-ucp/Quantum-Eye.git
cd Quantum-Eye
jupyter notebook Notebooks/multi-basis_bell.ipynb
```

### 1.3 Run the Experiment (5 minutes)
Within the notebook:

- Set `USE_HARDWARE = False` for simulation (no IBM account required).
- For IBM hardware:
    - Set `USE_HARDWARE = True`
    - Provide `IBM_TOKEN`, `IBM_INSTANCE`
    - Backend default: `ibm_brisbane`

**Press "Run All".**  
The notebook:

1. Prepares the Bell state.
2. Takes **256 Z-basis shots**.
3. Runs Quantum Eye frequency-domain pipeline.
4. Predicts cross-basis (X/Y) observables.
5. Takes **4096 X-basis hardware shots** for validation.
6. Computes observable with 95% CI (Hoeffding).
7. Logs to `validation_results/quantum_eye_validation_log.csv`.

---

## 2. Circuit and Observable

### 2.1 Bell Circuit (`bell_state_2x2`)

Standard 2-qubit Bell-state protocol:

1. Start in $\lvert 00 \rangle$
2. Apply $H$ to qubit 0
3. Apply CNOT (0 $\rightarrow$ 1)

Defined in the notebook; QASM export is trivial.

### 2.2 Observable (X-basis)

Define the binary random variable:

- $Y = 1$ if X-basis outcome $\in \{00, 11\}$
- $Y = 0$ if outcome $\in \{01, 10\}$

Observable:
$$
\langle O_X \rangle = P_X(00) + P_X(11)
$$

For ideal $\lvert \Phi^+ \rangle$, the expectation is $\mathbf{1.0}$.

---

## 3. Data Collection and Analysis

### A. Minimal-Shots Reconstruction (Quantum Eye)
- Circuit: 2-qubit Bell
- **256 Z-basis shots only**
- Extract four features:
    - $P$: phase-coherence proxy
    - $S$: inverse participation ratio
    - $E$: entropy-style feature
    - $Q$: correlation/entanglement proxy
- Features $\rightarrow$ $2 \times 2$ array $\rightarrow$ frequency-domain transform ("quantum fingerprint")
- Reconstructs state, predicts X/Y-basis observables

**Measurement Reduction:**
- Full tomography: $\sim$12,288 shots (3 bases × 4096)
- Quantum Eye: **256 Z-basis shots**

Reproducible in simulation.

### B. Direct Hardware Validation (Tracker Value)
- Backend: **ibm_brisbane**
- **4096 X-basis shots**
- Same Bell circuit, measured in X
- Empirical observable:
$$
\hat{\mu} = P_X(00) + P_X(11) = 0.961
$$

Result submitted: **0.961**

---

## 4. Error Bars (Hoeffding, 95% CI)

Treat each X-basis shot as Bernoulli ($Y_i \in \{0,1\}$).

- Number of shots: $N = 4096$
- Empirical mean: $\hat{\mu} = 0.961$

Hoeffding inequality:
$$
\Pr(|\hat{\mu}-\mu| \geq \varepsilon) \leq 2 e^{-2 N \varepsilon^2}
$$

For $\delta = 0.05$, solve for $\varepsilon$:
$$
\varepsilon = \sqrt{\frac{\ln(2/\delta)}{2N}} \approx 0.0212
$$

**Interval:**

- Lower bound: $0.961 - 0.0212 \approx 0.940$
- Upper bound: $0.961 + 0.0212 \approx 0.982$

**Reported 95% CI:** $\mathbf{[0.940,\, 0.982]}$

---

## 5. Ease of Validation

The notebook is minimal and self-contained:

- Dependencies: `qiskit`, `numpy`, `matplotlib`, `pandas`, `seaborn`
- Contains:
    - Circuit definition
    - Data collection
    - Reconstruction
    - Observable and error bars
- Configuration:
    - `USE_HARDWARE`
    - Backend name
    - Shot counts
- Automated CSV logging

**Core Reproducible Claim:**
> Running this notebook on a standard Bell circuit produces  
> $\langle O_X \rangle = 0.961$ with Hoeffding-tight 95% CI of $\mathbf{[0.940,\, 0.982]}$  
> using **4096 X-basis shots** on `ibm_brisbane`.

---

## 6. Behind the Scenes (Optional Context)

While the Bell-state notebook is simple and self-contained, it is part of a much larger experimental framework. The broader Quantum Eye work exercises frequency-domain methods on larger, more complex systems—including multi-qubit GHZ families, correlation reconstruction, and comparisons to advanced error-mitigation protocols.

- Larger experiments: multi-qubit holographic reconstructions, VQE pipelines, TREX mitigation benchmarks
- All experiments consistently highlight measurement efficiency and competitive error mitigation.

**None of this is required to validate the Bell-state observable in this document**. The Bell demo is intended as a transparent front door; more advanced quantum protocols and benchmarks are available for further evaluation.

---

## 7. Compute Resources

### Quantum
- IBM backend: `ibm_brisbane`
- Circuit: 2-qubit Bell
- 256 Z-basis shots (Quantum Eye reconstruction)
- 4096 X-basis shots (observable + error bars)

### Classical
- CPU (laptop/workstation)
- Python 3.8+
- Packages: `qiskit`, `qiskit-aer`, `numpy`, `matplotlib`, `pandas`, `seaborn`
- Notebook: `Notebooks/multi-basis_bell.ipynb`
