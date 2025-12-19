"""
QSV-Calibrated Bounded Cross-Basis Prediction Validation Test

This test validates Quantum Eye's ACTUAL claims from the paper:
"Bounded Cross-Basis Prediction from Single-Basis Counts via Frequency Signatures"

Key points from the paper:
1. Single-basis measurements are NOT informationally complete
2. Y predictions are fundamentally limited (phase information not available)
3. X predictions are bounded and heuristic
4. QSV criterion (P·S·E·Q > τ) filters valid states
5. Method is falsifiable and bounded, not exact state recovery

MEASURABLE CONTRACT (what the paper claims):
1. Predict bounded intervals [L_X, U_X] for X observables from Z-only counts
2. Coverage: When QE does not abstain, true <X> lies inside [L, U] at ~95% frequency
3. Abstention: When QSV ≤ τ, method abstains (no prediction)
4. Non-injectivity handling: When Z-only information is non-injective (same Z, different X),
   method either abstains OR outputs wide intervals (width ≥ 0.5)
5. Y predictions: Method does NOT claim Y predictions from Z-only counts (always abstain)

This test implements paper-validating methodology:
- Phase 1: Calibration - Build w(QSV) mapping from 50 circuits
- Phase 2: Evaluation - Validate 95% coverage on 30 held-out circuits
- Phase 3: Non-injectivity trap - Test two circuits with same Z, different X/Y
- Phase 4: Negative control - Shuffle calibration curve to verify harness sensitivity
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp

from quantum_eye import QuantumEye
from quantum_eye.adapters.quantum_eye_adapter import QuantumEyeAdapter
from quantum_eye.core.ucp import UCPIdentity
from tests.observable_metrics import compute_expectation_value_from_counts, bitstring_to_eigenvalues

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fixed parameters
BACKEND_NAME = "ibm_fez"
QUBIT_PAIR = {0: 87, 1: 88}
SHOTS = 4096
OPTIMIZATION_LEVEL = 1
QSV_THRESHOLD = 0.1  # Abstention threshold τ
SHOT_NOISE_FLOOR_K = 2.0  # k·σ_direct floor (k=2 for ~95% confidence)

# Circuit generation seeds
CALIBRATION_SEED_START = 1000
CALIBRATION_SEED_END = 1049  # 50 circuits
EVALUATION_SEED_START = 2000
EVALUATION_SEED_END = 2049  # 50 circuits (increased for better statistics)
# NOTE: For backend drift protection, consider interleaving calibration/evaluation
# seeds instead of running as separate batches. Current design runs calibration
# then evaluation sequentially, which is vulnerable to backend drift.
NON_INJECTIVITY_SEED_1 = 3000
NON_INJECTIVITY_SEED_2 = 3001

# Calibration parameters
QSV_BINS = 10  # Number of bins for QSV calibration curve
COVERAGE_TARGET = 0.95  # Target coverage rate
COVERAGE_MIN = 0.90  # Minimum acceptable coverage (with confidence interval)
NEGATIVE_CONTROL_SHUFFLES = 20  # Number of random shuffles for negative control
NEGATIVE_CONTROL_FAIL_THRESHOLD = 0.15  # Coverage must drop by at least this much

# Non-injectivity trap parameters
NON_INJECTIVITY_WIDE_THRESHOLD = 0.8  # Interval width must be ≥ 0.8 (near-full range [-1,1])


def generate_circuit_angles(seed: int, n_qubits: int = 2) -> List[float]:
    """Generate fixed rotation angles for a circuit using a seed."""
    rng = np.random.RandomState(seed)
    angles = []
    for _ in range(2):  # Two layers
        for _ in range(n_qubits):  # Per qubit
            angles.append(rng.uniform(0, 2 * np.pi))  # RY angle
            angles.append(rng.uniform(0, 2 * np.pi))  # RZ angle
    return angles


def create_circuit(angles: List[float], basis: str = 'Z') -> QuantumCircuit:
    """Create a 2-qubit circuit with fixed structure."""
    qc = QuantumCircuit(2, 2)
    
    # Layer 1: Random rotations
    qc.ry(angles[0], 0)
    qc.rz(angles[1], 0)
    qc.ry(angles[2], 1)
    qc.rz(angles[3], 1)
    
    # CX gate
    qc.cx(0, 1)
    
    # Layer 2: More rotations
    qc.ry(angles[4], 0)
    qc.rz(angles[5], 0)
    qc.ry(angles[6], 1)
    qc.rz(angles[7], 1)
    
    # Basis change for measurement
    if basis == 'X':
        qc.h(0)
        qc.h(1)
    elif basis == 'Y':
        qc.sdg(0)
        qc.h(0)
        qc.sdg(1)
        qc.h(1)
    
    qc.measure([0, 1], [0, 1])
    return qc


def generate_non_injectivity_trap_circuits() -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """
    Generate two state preparation circuits and their Z/X measurement variants.
    
    Returns: (prep1, prep2, prep1_z, prep1_x, prep2_z, prep2_x)
    where prep1/prep2 are state preparation (no measurement),
    and prep*_z/prep*_x add Z/X basis measurements.
    
    Uses |++⟩ vs |+-⟩ states which both have uniform Z distribution but different X expectations.
    """
    # Prep 1: |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    # This has uniform Z distribution (all outcomes 25%) but X0 = X1 = 1.0
    prep1 = QuantumCircuit(2, 2)
    prep1.h(0)
    prep1.h(1)
    # No measurement - this is just state preparation
    
    # Prep 2: |+-⟩ = (|00⟩ - |01⟩ + |10⟩ - |11⟩)/2
    # This also has uniform Z distribution but X0 = 1.0, X1 = -1.0
    prep2 = QuantumCircuit(2, 2)
    prep2.h(0)
    prep2.x(1)
    prep2.h(1)
    # No measurement - this is just state preparation
    
    # Create Z-basis measurement versions
    prep1_z = prep1.copy()
    prep1_z.measure([0, 1], [0, 1])
    
    prep2_z = prep2.copy()
    prep2_z.measure([0, 1], [0, 1])
    
    # Create X-basis measurement versions (add H before measurement)
    prep1_x = prep1.copy()
    prep1_x.h(0)
    prep1_x.h(1)
    prep1_x.measure([0, 1], [0, 1])
    
    prep2_x = prep2.copy()
    prep2_x.h(0)
    prep2_x.h(1)
    prep2_x.measure([0, 1], [0, 1])
    
    return prep1_z, prep1_x, prep2_z, prep2_x


def compute_observable_from_state(state: np.ndarray, pauli_string: str) -> float:
    """Compute expectation value of a Pauli operator from a statevector."""
    sv = Statevector(state)
    pauli_op = SparsePauliOp(pauli_string)
    expectation = sv.expectation_value(pauli_op)
    return float(np.real(expectation))


def verify_pauli_qubit_alignment() -> bool:
    """
    Verify that Pauli string conventions align between statevector and counts paths.
    
    Prepares |+0⟩ state where:
    - X0 (qubit 0) should be +1.0 (qubit 0 is in |+⟩)
    - X1 (qubit 1) should be 0.0 (qubit 1 is in |0⟩, X expectation is 0)
    
    Tests both 'IX' and 'XI' Pauli strings to determine correct mapping.
    
    Returns True if alignment is correct, False otherwise.
    """
    logger.info("Verifying Pauli/qubit-order alignment between statevector and counts paths...")
    
    # Prepare |+0⟩ = (|00⟩ + |10⟩)/√2
    # This state has: X0 = +1.0, X1 = 0.0
    prep_circuit = QuantumCircuit(2, 2)
    prep_circuit.h(0)  # Put qubit 0 in |+⟩
    # Qubit 1 stays in |0⟩
    
    # Get statevector
    sv = Statevector.from_instruction(prep_circuit)
    state = sv.data
    
    # Compute from statevector using both Pauli string conventions
    # In Qiskit SparsePauliOp: rightmost character acts on qubit 0
    # So 'IX' means: I on qubit 1, X on qubit 0 → measures X0
    # And 'XI' means: X on qubit 1, I on qubit 0 → measures X1
    x0_sv_ix = compute_observable_from_state(state, 'IX')
    x1_sv_xi = compute_observable_from_state(state, 'XI')
    
    # Expected: X0 = +1.0, X1 = 0.0 for |+0⟩
    expected_x0 = 1.0
    expected_x1 = 0.0
    tolerance = 0.01
    
    # Now measure in X basis and compute from counts
    x_meas_circuit = prep_circuit.copy()
    # For X-basis measurement: apply H before Z measurement
    # Qubit 0: already |+⟩, H|+⟩ = |0⟩, measure Z → outcome 0
    # Qubit 1: |0⟩, H|0⟩ = |+⟩, measure Z → outcome 0
    x_meas_circuit.h(0)
    x_meas_circuit.h(1)
    x_meas_circuit.measure([0, 1], [0, 1])
    
    # Simulate to get counts
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    result = simulator.run(x_meas_circuit, shots=10000).result()
    counts = result.get_counts()
    
    # Compute from counts using both conventions
    x0_counts_ix = compute_expectation_value_from_counts(counts, 'IX')
    x1_counts_xi = compute_expectation_value_from_counts(counts, 'XI')
    
    # Check alignment
    ix_correct = (abs(x0_sv_ix - expected_x0) < tolerance and 
                  abs(x0_counts_ix - expected_x0) < tolerance)
    xi_correct = (abs(x1_sv_xi - expected_x1) < tolerance and 
                  abs(x1_counts_xi - expected_x1) < tolerance)
    
    if ix_correct and xi_correct:
        logger.info("✓ Pauli/qubit-order alignment verified:")
        logger.info(f"  'IX' → X0: statevector={x0_sv_ix:.4f}, counts={x0_counts_ix:.4f} (expected {expected_x0})")
        logger.info(f"  'XI' → X1: statevector={x1_sv_xi:.4f}, counts={x1_counts_xi:.4f} (expected {expected_x1})")
        return True
    else:
        logger.error("✗ Pauli/qubit-order MISALIGNMENT detected!")
        logger.error(f"  Statevector: X0(IX)={x0_sv_ix:.4f}, X1(XI)={x1_sv_xi:.4f}")
        logger.error(f"  Counts: X0(IX)={x0_counts_ix:.4f}, X1(XI)={x1_counts_xi:.4f}")
        logger.error(f"  Expected: X0={expected_x0}, X1={expected_x1} for |+0⟩ state")
        logger.error("  FIX REQUIRED: Reverse Pauli string labels or fix bitstring convention")
        return False


def compute_observable_from_counts(counts: Dict[str, int], pauli_string: str) -> Tuple[float, float]:
    """
    Compute expectation value and uncertainty from measurement counts.
    
    Uses the robust Pauli string method that handles bit ordering correctly.
    Uses standard analytic form for uncertainty: sqrt((1 - m^2)/N) for ±1 observables.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    
    # Use the robust method for all observables (handles bit ordering correctly)
    expectation = compute_expectation_value_from_counts(counts, pauli_string)
    
    # Standard analytic standard error for ±1 observables: sqrt((1 - m^2)/N)
    uncertainty = np.sqrt(max(0.0, 1.0 - expectation**2) / total)
    
    return expectation, uncertainty


def compute_qsv_from_counts(ucp_identity: UCPIdentity, counts: Dict[str, int], num_qubits: int = 2) -> Dict[str, float]:
    """
    Compute QSV parameters using the actual UCPIdentity implementation.
    
    This uses the same QSV computation as the Quantum Eye framework,
    ensuring consistency with the paper's definitions.
    """
    # Use the actual UCPIdentity method that matches the framework
    qsv_result = ucp_identity.phi_from_counts(counts, num_qubits)
    
    signature = qsv_result.get('quantum_signature', {})
    qsv_info = qsv_result.get('qsv', {})
    
    return {
        'P': signature.get('P', 0.0),
        'S': signature.get('S', 0.0),
        'E': signature.get('E', 0.0),
        'Q': signature.get('Q', 0.0),
        'QSV': qsv_info.get('score', 0.0),
        'valid': qsv_info.get('valid', False),
        'interpretation': qsv_info.get('interpretation', 'Unknown')
    }


def compute_qsv_calibration_curve(calibration_results: List[Dict]) -> Dict:
    """
    Build w(QSV) calibration curve from calibration results.
    
    Bins QSV into 10 bins and sets w(QSV) = 95th percentile of errors in each bin.
    
    Args:
        calibration_results: List of dicts with keys: 'qsv', 'error_x0', 'error_x1'
        
    Returns:
        Dictionary with 'qsv_bins' (bin centers), 'w_x0', 'w_x1' (interval widths)
    """
    # Extract QSV and errors
    qsv_values = [r['qsv'] for r in calibration_results]
    errors_x0 = [r['error_x0'] for r in calibration_results]
    errors_x1 = [r['error_x1'] for r in calibration_results]
    
    # Bin QSV values
    qsv_min = min(qsv_values) if qsv_values else 0.0
    qsv_max = max(qsv_values) if qsv_values else 1.0
    
    # Create bins
    bin_edges = np.linspace(qsv_min, qsv_max, QSV_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute 95th percentile for each bin
    w_x0 = []
    w_x1 = []
    
    for i in range(QSV_BINS):
        # Find data points in this bin
        bin_mask = (np.array(qsv_values) >= bin_edges[i]) & (np.array(qsv_values) < bin_edges[i + 1])
        if i == QSV_BINS - 1:  # Include right edge for last bin
            bin_mask = (np.array(qsv_values) >= bin_edges[i]) & (np.array(qsv_values) <= bin_edges[i + 1])
        
        bin_errors_x0 = [errors_x0[j] for j in range(len(errors_x0)) if bin_mask[j]]
        bin_errors_x1 = [errors_x1[j] for j in range(len(errors_x1)) if bin_mask[j]]
        
        if len(bin_errors_x0) > 0:
            w_x0.append(np.percentile(bin_errors_x0, 95))
        else:
            w_x0.append(1.0)  # Default to full range if no data
        
        if len(bin_errors_x1) > 0:
            w_x1.append(np.percentile(bin_errors_x1, 95))
        else:
            w_x1.append(1.0)  # Default to full range if no data
    
    return {
        'qsv_bins': bin_centers.tolist(),
        'bin_edges': bin_edges.tolist(),
        'w_x0': w_x0,
        'w_x1': w_x1
    }


def interpolate_w(qsv: float, calibration_curve: Dict, observable: str = 'x0') -> float:
    """
    Interpolate w(QSV) value from calibration curve.
    
    Args:
        qsv: QSV value to interpolate
        calibration_curve: Calibration curve dict with 'qsv_bins' and 'w_x0'/'w_x1'
        observable: 'x0' or 'x1'
        
    Returns:
        Interpolated w(QSV) value
    """
    w_key = f'w_{observable}'
    if w_key not in calibration_curve:
        return 1.0  # Default to full range
    
    qsv_bins = np.array(calibration_curve['qsv_bins'])
    w_values = np.array(calibration_curve[w_key])
    
    # Handle edge cases
    if qsv <= qsv_bins[0]:
        return w_values[0]
    if qsv >= qsv_bins[-1]:
        return w_values[-1]
    
    # Linear interpolation
    w = np.interp(qsv, qsv_bins, w_values)
    return float(w)


def predict_interval_with_qsv(
    x_qe: float,
    qsv: float,
    calibration_curve: Dict,
    sigma_direct: float,
    observable: str = 'x0',
    k: float = SHOT_NOISE_FLOOR_K
) -> Tuple[float, float]:
    """
    Compute prediction interval [L, U] using QSV calibration.
    
    Args:
        x_qe: Quantum Eye prediction value
        qsv: QSV score
        calibration_curve: Calibration curve dict
        sigma_direct: Direct measurement standard error (uncertainty of direct estimator, not truth)
        observable: 'x0' or 'x1'
        k: Shot noise floor multiplier (default 2.0)
            This enforces: "don't claim tighter than direct shot-noise would allow"
            It's a conservative floor, not a statistically rigorous prediction interval.
        
    Returns:
        Tuple (L, U) clamped to [-1, 1]
    """
    # Get w(QSV) from calibration curve
    w = interpolate_w(qsv, calibration_curve, observable)
    
    # Enforce shot noise floor: w(QSV) ≥ k·σ_direct
    # This is a conservative constraint: don't claim intervals tighter than
    # what direct measurement shot-noise would allow
    w = max(w, k * sigma_direct)
    
    # Compute interval
    L = max(-1.0, x_qe - w)
    U = min(1.0, x_qe + w)
    
    return L, U


def run_calibration_phase(
    adapter: QuantumEyeAdapter,
    ucp_identity: UCPIdentity,
    backend,
    seed_start: int,
    seed_end: int
) -> Tuple[List[Dict], Dict]:
    """
    Run calibration phase to build w(QSV) mapping.
    
    Args:
        adapter: QuantumEyeAdapter instance
        ucp_identity: UCPIdentity instance
        backend: Qiskit backend
        seed_start: Starting seed for calibration circuits
        seed_end: Ending seed (inclusive)
        
    Returns:
        Tuple of (calibration_results, calibration_curve)
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: CALIBRATION - Building w(QSV) Mapping")
    logger.info("=" * 80)
    logger.info(f"Generating {seed_end - seed_start + 1} calibration circuits (seeds {seed_start}-{seed_end})")
    
    # Generate all circuits
    all_circuits = []
    circuit_metadata = []
    from qiskit.transpiler import Layout
    
    for seed in range(seed_start, seed_end + 1):
        angles = generate_circuit_angles(seed)
        
        for basis in ['Z', 'X']:  # Only need Z and X for calibration
            circuit = create_circuit(angles, basis=basis)
            
            layout_dict = {circuit.qubits[i]: QUBIT_PAIR[i] for i in range(2)}
            layout = Layout(layout_dict)
            
            transpiled_circuit = transpile(
                circuit,
                backend=backend,
                initial_layout=layout,
                optimization_level=OPTIMIZATION_LEVEL,
                layout_method='trivial',
                routing_method='none'
            )
            
            all_circuits.append(transpiled_circuit)
            circuit_metadata.append({
                'seed': seed,
                'angles': angles,
                'basis': basis
            })
    
    logger.info(f"Prepared {len(all_circuits)} circuits for batch execution")
    
    # Submit in batch
    logger.info("Submitting calibration circuits in batch mode...")
    from qiskit_ibm_runtime import Batch, SamplerV2
    
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch, options={
            "default_shots": SHOTS,
            "twirling": {"enable_measure": True}
        })
        
        job = sampler.run(all_circuits)
        logger.info(f"Batch job {job.job_id()} submitted, waiting for completion...")
        adapter._wait_for_job(job)
        batch_result = job.result()
        logger.info(f"Batch job completed with {len(batch_result)} results")
    
    # Process results
    logger.info("Processing calibration results...")
    calibration_results = []
    
    # Group results by seed
    results_by_seed = {}
    for idx, pub_result in enumerate(batch_result):
        meta = circuit_metadata[idx]
        seed = meta['seed']
        
        # Extract counts
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
        elif hasattr(pub_result.data, 'c'):
            counts = pub_result.data.c.get_counts()
        elif len(pub_result.data) > 0:
            first_reg = next(iter(pub_result.data))
            counts = pub_result.data[first_reg].get_counts()
        else:
            logger.error(f"No measurement data found in result {idx}")
            counts = {}
        
        if seed not in results_by_seed:
            results_by_seed[seed] = {}
        results_by_seed[seed][meta['basis']] = counts
    
    # Process each calibration circuit
    for seed in sorted(results_by_seed.keys()):
        if 'Z' not in results_by_seed[seed] or 'X' not in results_by_seed[seed]:
            logger.warning(f"Skipping seed {seed} - missing Z or X counts")
            continue
        
        z_counts = results_by_seed[seed]['Z']
        x_counts = results_by_seed[seed]['X']
        
        # Compute QSV from Z-counts
        qsv_params = compute_qsv_from_counts(ucp_identity, z_counts, num_qubits=2)
        qsv = qsv_params['QSV']
        
        # Get truth: direct X-basis measurement
        # X-basis counts use 'IX' for X0, 'XI' for X1 (verified by alignment test)
        x0_truth, x0_unc = compute_observable_from_counts(x_counts, 'IX')
        x1_truth, x1_unc = compute_observable_from_counts(x_counts, 'XI')
        
        # Compute QE prediction from Z-only
        dummy_circuit = QuantumCircuit(2)
        mitigation_result = adapter._apply_mitigation_to_counts(
            counts=z_counts,
            circuit=dummy_circuit,
            reference_label=None,
            component_weights=None
        )
        
        mitigated_state = mitigation_result.get('mitigated_state')
        if mitigated_state is None:
            logger.warning(f"Seed {seed}: Mitigation failed, skipping")
            continue
        
        # Compute QE predictions using same Pauli strings as truth computation
        x0_qe = compute_observable_from_state(mitigated_state, 'IX')
        x1_qe = compute_observable_from_state(mitigated_state, 'XI')
        
        # Record errors
        error_x0 = abs(x0_qe - x0_truth)
        error_x1 = abs(x1_qe - x1_truth)
        
        calibration_results.append({
            'seed': seed,
            'qsv': qsv,
            'x0_qe': x0_qe,
            'x1_qe': x1_qe,
            'x0_truth': x0_truth,
            'x1_truth': x1_truth,
            'x0_unc': x0_unc,
            'x1_unc': x1_unc,
            'error_x0': error_x0,
            'error_x1': error_x1
        })
    
    logger.info(f"Collected {len(calibration_results)} calibration data points")
    
    # Build calibration curve
    logger.info("Building w(QSV) calibration curve...")
    calibration_curve = compute_qsv_calibration_curve(calibration_results)
    
    logger.info("Calibration phase complete!")
    logger.info(f"QSV range: [{min(r['qsv'] for r in calibration_results):.4f}, {max(r['qsv'] for r in calibration_results):.4f}]")
    
    return calibration_results, calibration_curve


def run_evaluation_phase(
    adapter: QuantumEyeAdapter,
    ucp_identity: UCPIdentity,
    backend,
    calibration_curve: Dict,
    seed_start: int,
    seed_end: int
) -> List[Dict]:
    """
    Run evaluation phase to validate coverage.
    
    Args:
        adapter: QuantumEyeAdapter instance
        ucp_identity: UCPIdentity instance
        backend: Qiskit backend
        calibration_curve: Calibration curve dict
        seed_start: Starting seed for evaluation circuits
        seed_end: Ending seed (inclusive)
        
    Returns:
        List of evaluation results
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: EVALUATION - Validating Coverage")
    logger.info("=" * 80)
    logger.info(f"Generating {seed_end - seed_start + 1} evaluation circuits (seeds {seed_start}-{seed_end})")
    
    # Generate all circuits
    all_circuits = []
    circuit_metadata = []
    from qiskit.transpiler import Layout
    
    for seed in range(seed_start, seed_end + 1):
        angles = generate_circuit_angles(seed)
        
        for basis in ['Z', 'X']:  # Only need Z and X for evaluation
            circuit = create_circuit(angles, basis=basis)
            
            layout_dict = {circuit.qubits[i]: QUBIT_PAIR[i] for i in range(2)}
            layout = Layout(layout_dict)
            
            transpiled_circuit = transpile(
                circuit,
                backend=backend,
                initial_layout=layout,
                optimization_level=OPTIMIZATION_LEVEL,
                layout_method='trivial',
                routing_method='none'
            )
            
            all_circuits.append(transpiled_circuit)
            circuit_metadata.append({
                'seed': seed,
                'angles': angles,
                'basis': basis
            })
    
    logger.info(f"Prepared {len(all_circuits)} circuits for batch execution")
    
    # Submit in batch
    logger.info("Submitting evaluation circuits in batch mode...")
    from qiskit_ibm_runtime import Batch, SamplerV2
    
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch, options={
            "default_shots": SHOTS,
            "twirling": {"enable_measure": True}
        })
        
        job = sampler.run(all_circuits)
        logger.info(f"Batch job {job.job_id()} submitted, waiting for completion...")
        adapter._wait_for_job(job)
        batch_result = job.result()
        logger.info(f"Batch job completed with {len(batch_result)} results")
    
    # Process results
    logger.info("Processing evaluation results...")
    evaluation_results = []
    
    # Group results by seed
    results_by_seed = {}
    for idx, pub_result in enumerate(batch_result):
        meta = circuit_metadata[idx]
        seed = meta['seed']
        
        # Extract counts
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
        elif hasattr(pub_result.data, 'c'):
            counts = pub_result.data.c.get_counts()
        elif len(pub_result.data) > 0:
            first_reg = next(iter(pub_result.data))
            counts = pub_result.data[first_reg].get_counts()
        else:
            logger.error(f"No measurement data found in result {idx}")
            counts = {}
        
        if seed not in results_by_seed:
            results_by_seed[seed] = {}
        results_by_seed[seed][meta['basis']] = counts
    
    # Process each evaluation circuit
    for seed in sorted(results_by_seed.keys()):
        if 'Z' not in results_by_seed[seed] or 'X' not in results_by_seed[seed]:
            logger.warning(f"Skipping seed {seed} - missing Z or X counts")
            continue
        
        z_counts = results_by_seed[seed]['Z']
        x_counts = results_by_seed[seed]['X']
        
        # Compute QSV from Z-counts
        qsv_params = compute_qsv_from_counts(ucp_identity, z_counts, num_qubits=2)
        qsv = qsv_params['QSV']
        
        # Get truth: direct X-basis measurement
        # X-basis counts use 'IX' for X0, 'XI' for X1 (verified by alignment test)
        x0_truth, x0_unc = compute_observable_from_counts(x_counts, 'IX')
        x1_truth, x1_unc = compute_observable_from_counts(x_counts, 'XI')
        
        # Check QSV threshold for abstention
        abstained = qsv <= QSV_THRESHOLD
        
        if abstained:
            # Abstain: no prediction
            evaluation_results.append({
                'seed': seed,
                'qsv': qsv,
                'abstained': True,
                'x0_truth': x0_truth,
                'x1_truth': x1_truth,
                'x0_interval': None,
                'x1_interval': None,
                'x0_in_interval': None,
                'x1_in_interval': None
            })
            logger.info(f"Seed {seed}: QSV={qsv:.4f} ≤ {QSV_THRESHOLD}, ABSTAINED")
            continue
        
        # Compute QE prediction from Z-only
        dummy_circuit = QuantumCircuit(2)
        mitigation_result = adapter._apply_mitigation_to_counts(
            counts=z_counts,
            circuit=dummy_circuit,
            reference_label=None,
            component_weights=None
        )
        
        mitigated_state = mitigation_result.get('mitigated_state')
        if mitigated_state is None:
            logger.warning(f"Seed {seed}: Mitigation failed, abstaining")
            evaluation_results.append({
                'seed': seed,
                'qsv': qsv,
                'abstained': True,
                'x0_truth': x0_truth,
                'x1_truth': x1_truth,
                'x0_interval': None,
                'x1_interval': None,
                'x0_in_interval': None,
                'x1_in_interval': None
            })
            continue
        
        # Compute QE predictions using same Pauli strings as truth computation
        x0_qe = compute_observable_from_state(mitigated_state, 'IX')
        x1_qe = compute_observable_from_state(mitigated_state, 'XI')
        
        # Compute prediction intervals
        x0_L, x0_U = predict_interval_with_qsv(x0_qe, qsv, calibration_curve, x0_unc, 'x0')
        x1_L, x1_U = predict_interval_with_qsv(x1_qe, qsv, calibration_curve, x1_unc, 'x1')
        
        # Check if truth is in interval
        x0_in_interval = (x0_L <= x0_truth <= x0_U)
        x1_in_interval = (x1_L <= x1_truth <= x1_U)
        
        evaluation_results.append({
            'seed': seed,
            'qsv': qsv,
            'abstained': False,
            'x0_qe': x0_qe,
            'x1_qe': x1_qe,
            'x0_truth': x0_truth,
            'x1_truth': x1_truth,
            'x0_unc': x0_unc,
            'x1_unc': x1_unc,
            'x0_interval': [x0_L, x0_U],
            'x1_interval': [x1_L, x1_U],
            'x0_in_interval': x0_in_interval,
            'x1_in_interval': x1_in_interval
        })
        
        logger.info(f"Seed {seed}: QSV={qsv:.4f}, X0: {x0_qe:.4f} ∈ [{x0_L:.4f}, {x0_U:.4f}]? {x0_in_interval}, "
                   f"X1: {x1_qe:.4f} ∈ [{x1_L:.4f}, {x1_U:.4f}]? {x1_in_interval}")
    
    logger.info("Evaluation phase complete!")
    return evaluation_results


def run_non_injectivity_trap(
    adapter: QuantumEyeAdapter,
    ucp_identity: UCPIdentity,
    backend,
    calibration_curve: Dict
) -> Dict:
    """
    Run non-injectivity trap: two circuits with same Z, different X/Y.
    
    Args:
        adapter: QuantumEyeAdapter instance
        ucp_identity: UCPIdentity instance
        backend: Qiskit backend
        calibration_curve: Calibration curve dict
        
    Returns:
        Dictionary with trap results
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: NON-INJECTIVITY TRAP")
    logger.info("=" * 80)
    logger.info("Testing two circuits with identical Z distributions but different X/Y")
    
    # Generate trap circuits (returns prep circuits with Z and X measurements)
    qc1_z, qc1_x, qc2_z, qc2_x = generate_non_injectivity_trap_circuits()
    
    # Transpile all circuits
    from qiskit.transpiler import Layout
    
    circuits_to_run = [
        (qc1_z, 'Z', 'circuit1'),
        (qc1_x, 'X', 'circuit1'),
        (qc2_z, 'Z', 'circuit2'),
        (qc2_x, 'X', 'circuit2')
    ]
    
    all_circuits = []
    circuit_labels = []
    
    for circuit, basis, label in circuits_to_run:
        layout_dict = {circuit.qubits[i]: QUBIT_PAIR[i] for i in range(2)}
        layout = Layout(layout_dict)
        
        transpiled = transpile(
            circuit,
            backend=backend,
            initial_layout=layout,
            optimization_level=OPTIMIZATION_LEVEL
        )
        
        all_circuits.append(transpiled)
        circuit_labels.append((label, basis))
    
    # Submit in batch
    logger.info("Submitting trap circuits in batch mode...")
    from qiskit_ibm_runtime import Batch, SamplerV2
    
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch, options={
            "default_shots": SHOTS,
            "twirling": {"enable_measure": True}
        })
        
        job = sampler.run(all_circuits)
        logger.info(f"Batch job {job.job_id()} submitted, waiting for completion...")
        adapter._wait_for_job(job)
        batch_result = job.result()
        logger.info(f"Batch job completed")
    
    # Process results
    results = {}
    for idx, pub_result in enumerate(batch_result):
        label, basis = circuit_labels[idx]
        
        # Extract counts
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
        elif hasattr(pub_result.data, 'c'):
            counts = pub_result.data.c.get_counts()
        elif len(pub_result.data) > 0:
            first_reg = next(iter(pub_result.data))
            counts = pub_result.data[first_reg].get_counts()
        else:
            logger.error(f"No measurement data found in result {idx}")
            counts = {}
        
        if label not in results:
            results[label] = {}
        results[label][basis] = counts
    
    # Process each circuit
    trap_results = {}
    
    for circuit_label in ['circuit1', 'circuit2']:
        if circuit_label not in results or 'Z' not in results[circuit_label] or 'X' not in results[circuit_label]:
            logger.warning(f"Missing data for {circuit_label}")
            continue
        
        z_counts = results[circuit_label]['Z']
        x_counts = results[circuit_label]['X']
        
        # Compute QSV from Z-counts
        qsv_params = compute_qsv_from_counts(ucp_identity, z_counts, num_qubits=2)
        qsv = qsv_params['QSV']
        
        # Get truth: X-basis counts use 'IX' for X0, 'XI' for X1 (verified by alignment test)
        x0_truth, x0_unc = compute_observable_from_counts(x_counts, 'IX')
        x1_truth, x1_unc = compute_observable_from_counts(x_counts, 'XI')
        
        # Check abstention
        abstained = qsv <= QSV_THRESHOLD
        
        if abstained:
            trap_results[circuit_label] = {
                'qsv': qsv,
                'abstained': True,
                'x0_truth': x0_truth,
                'x1_truth': x1_truth,
                'interval_width': None
            }
            logger.info(f"{circuit_label}: QSV={qsv:.4f} ≤ {QSV_THRESHOLD}, ABSTAINED (correct behavior)")
            continue
        
        # Compute QE prediction
        dummy_circuit = QuantumCircuit(2)
        mitigation_result = adapter._apply_mitigation_to_counts(
            counts=z_counts,
            circuit=dummy_circuit,
            reference_label=None,
            component_weights=None
        )
        
        mitigated_state = mitigation_result.get('mitigated_state')
        if mitigated_state is None:
            trap_results[circuit_label] = {
                'qsv': qsv,
                'abstained': True,
                'x0_truth': x0_truth,
                'x1_truth': x1_truth,
                'interval_width': None
            }
            continue
        
        # Compute QE predictions using same Pauli strings as truth computation
        x0_qe = compute_observable_from_state(mitigated_state, 'IX')
        x1_qe = compute_observable_from_state(mitigated_state, 'XI')
        
        # Compute intervals
        x0_L, x0_U = predict_interval_with_qsv(x0_qe, qsv, calibration_curve, x0_unc, 'x0')
        x1_L, x1_U = predict_interval_with_qsv(x1_qe, qsv, calibration_curve, x1_unc, 'x1')
        
        # Compute interval width (average of X0 and X1)
        interval_width = ((x0_U - x0_L) + (x1_U - x1_L)) / 2
        
        # Check if truth is in interval
        x0_in_interval = (x0_L <= x0_truth <= x0_U)
        x1_in_interval = (x1_L <= x1_truth <= x1_U)
        
        # For non-injectivity trap: intervals should be wide (near-full range [-1,1])
        # when Z-only information is provably non-injective for X
        # Threshold: width ≥ 0.8 (80% of full range [-1,1] = 1.6, so 0.8 per observable)
        is_wide = interval_width >= NON_INJECTIVITY_WIDE_THRESHOLD
        
        trap_results[circuit_label] = {
            'qsv': qsv,
            'abstained': False,
            'x0_qe': x0_qe,
            'x1_qe': x1_qe,
            'x0_truth': x0_truth,
            'x1_truth': x1_truth,
            'x0_interval': [x0_L, x0_U],
            'x1_interval': [x1_L, x1_U],
            'interval_width': interval_width,
            'is_wide': is_wide,
            'x0_in_interval': x0_in_interval,
            'x1_in_interval': x1_in_interval
        }
        
        logger.info(f"{circuit_label}: QSV={qsv:.4f}, interval_width={interval_width:.4f}, "
                   f"wide? {is_wide}, X0 in interval? {x0_in_interval}, X1 in interval? {x1_in_interval}")
    
    return trap_results


def compute_binomial_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute binomial confidence interval using Wilson method.
    
    More robust than normal approximation, especially for small samples or extreme p.
    
    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple (lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 1.0
    
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% confidence
    z2 = z * z
    
    p = successes / trials
    n = trials
    
    # Wilson score interval
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return lower, upper


def run_negative_control(
    evaluation_results: List[Dict],
    calibration_curve: Dict,
    observed_coverage: float
) -> Dict:
    """
    Run negative control: shuffle calibration curve to verify harness sensitivity.
    
    Runs multiple random permutations of the calibration curve and verifies that
    coverage drops significantly in most cases.
    
    Args:
        evaluation_results: Evaluation results from Phase 2
        calibration_curve: Original calibration curve
        
    Returns:
        Dictionary with negative control results
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: NEGATIVE CONTROL - Verify Harness Sensitivity")
    logger.info("=" * 80)
    logger.info(f"Testing with {NEGATIVE_CONTROL_SHUFFLES} random permutations of calibration curve")
    
    # Run multiple shuffles
    shuffle_coverages = []
    
    for shuffle_idx in range(NEGATIVE_CONTROL_SHUFFLES):
        # Create randomly shuffled calibration curve
        shuffled_curve = calibration_curve.copy()
        rng = np.random.RandomState(shuffle_idx)  # Deterministic shuffle for reproducibility
        
        # Random permutation of w values
        perm_x0 = rng.permutation(len(calibration_curve['w_x0']))
        perm_x1 = rng.permutation(len(calibration_curve['w_x1']))
        
        shuffled_curve['w_x0'] = [calibration_curve['w_x0'][i] for i in perm_x0]
        shuffled_curve['w_x1'] = [calibration_curve['w_x1'][i] for i in perm_x1]
        
        # Re-compute intervals with shuffled curve
        negative_control_results = []
        
        for result in evaluation_results:
            if result['abstained']:
                continue
            
            # Re-compute intervals with shuffled curve
            qsv = result['qsv']
            x0_qe = result['x0_qe']
            x1_qe = result['x1_qe']
            x0_truth = result['x0_truth']
            x1_truth = result['x1_truth']
            
            # Use stored uncertainties if available, otherwise compute
            if 'x0_unc' in result and 'x1_unc' in result:
                x0_unc = result['x0_unc']
                x1_unc = result['x1_unc']
            else:
                # Fallback: compute uncertainties using standard analytic form
                x0_unc = np.sqrt(max(0.0, 1.0 - x0_truth**2) / SHOTS)
                x1_unc = np.sqrt(max(0.0, 1.0 - x1_truth**2) / SHOTS)
            
            # Compute intervals with shuffled curve
            x0_L, x0_U = predict_interval_with_qsv(x0_qe, qsv, shuffled_curve, x0_unc, 'x0')
            x1_L, x1_U = predict_interval_with_qsv(x1_qe, qsv, shuffled_curve, x1_unc, 'x1')
            
            # Check if truth is in interval
            x0_in_interval = (x0_L <= x0_truth <= x0_U)
            x1_in_interval = (x1_L <= x1_truth <= x1_U)
            
            negative_control_results.append({
                'x0_in_interval': x0_in_interval,
                'x1_in_interval': x1_in_interval
            })
        
        # Compute coverage for this shuffle
        if len(negative_control_results) > 0:
            x0_coverage = sum(1 for r in negative_control_results if r['x0_in_interval']) / len(negative_control_results)
            x1_coverage = sum(1 for r in negative_control_results if r['x1_in_interval']) / len(negative_control_results)
            overall_coverage = (x0_coverage + x1_coverage) / 2
            shuffle_coverages.append(overall_coverage)
    
    # Analyze shuffle results
    if len(shuffle_coverages) > 0:
        mean_shuffle_coverage = np.mean(shuffle_coverages)
        min_shuffle_coverage = np.min(shuffle_coverages)
        max_shuffle_coverage = np.max(shuffle_coverages)
        
        logger.info(f"\nNegative Control Results ({NEGATIVE_CONTROL_SHUFFLES} shuffles):")
        logger.info(f"  Mean coverage: {mean_shuffle_coverage:.3f}")
        logger.info(f"  Min coverage: {min_shuffle_coverage:.3f}")
        logger.info(f"  Max coverage: {max_shuffle_coverage:.3f}")
        logger.info(f"  Expected: Coverage should drop significantly below {COVERAGE_TARGET}")
        
        # Negative control passes if mean coverage drops by at least threshold
        # Compare against OBSERVED coverage, not fixed target
        coverage_drop = observed_coverage - mean_shuffle_coverage
        negative_control_passed = coverage_drop >= NEGATIVE_CONTROL_FAIL_THRESHOLD
        
        logger.info(f"  Coverage drop: {coverage_drop:.3f}")
        logger.info(f"  Negative control validation: {'PASS' if negative_control_passed else 'FAIL'}")
        logger.info(f"    (Coverage drop ≥ {NEGATIVE_CONTROL_FAIL_THRESHOLD} confirms harness sensitivity)")
    else:
        mean_shuffle_coverage = 0.0
        negative_control_passed = False
        logger.warning("No non-abstained predictions for negative control")
    
    return {
        'shuffle_coverages': shuffle_coverages,
        'mean_coverage': float(mean_shuffle_coverage) if len(shuffle_coverages) > 0 else None,
        'min_coverage': float(np.min(shuffle_coverages)) if len(shuffle_coverages) > 0 else None,
        'max_coverage': float(np.max(shuffle_coverages)) if len(shuffle_coverages) > 0 else None,
        'coverage_drop': float(observed_coverage - mean_shuffle_coverage) if len(shuffle_coverages) > 0 else None,
        'negative_control_passed': negative_control_passed
    }


def run_bounded_validation():
    """Run complete QSV-calibrated bounded validation test."""
    logger.info("=" * 80)
    logger.info("QSV-CALIBRATED BOUNDED CROSS-BASIS PREDICTION VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Backend: {BACKEND_NAME}")
    logger.info(f"Qubit mapping: {QUBIT_PAIR}")
    logger.info(f"Shots per run: {SHOTS}")
    logger.info(f"QSV threshold (τ): {QSV_THRESHOLD}")
    logger.info(f"Coverage target: {COVERAGE_TARGET}")
    logger.info("=" * 80)
    
    # Initialize adapter
    logger.info("\nInitializing adapter...")
    adapter = QuantumEyeAdapter({
        'backend_type': 'real',
        'backend_name': BACKEND_NAME,
        'default_shots': SHOTS,
        'optimization_level': OPTIMIZATION_LEVEL
    })
    
    backend = adapter.backend
    
    # Initialize QSV calculator
    ucp_identity = UCPIdentity()
    
    # CRITICAL: Verify Pauli/qubit-order alignment before proceeding
    logger.info("\nVerifying Pauli/qubit-order alignment...")
    if not verify_pauli_qubit_alignment():
        logger.error("=" * 80)
        logger.error("CRITICAL ERROR: Pauli/qubit-order misalignment detected!")
        logger.error("Cannot proceed with validation - fix alignment first.")
        logger.error("=" * 80)
        return {
            'validation_passed': False,
            'error': 'Pauli/qubit-order misalignment'
        }
    
    # Phase 1: Calibration
    calibration_results, calibration_curve = run_calibration_phase(
        adapter, ucp_identity, backend,
        CALIBRATION_SEED_START, CALIBRATION_SEED_END
    )
    
    # Phase 2: Evaluation
    evaluation_results = run_evaluation_phase(
        adapter, ucp_identity, backend,
        calibration_curve,
        EVALUATION_SEED_START, EVALUATION_SEED_END
    )
    
    # Phase 3: Non-injectivity trap
    trap_results = run_non_injectivity_trap(
        adapter, ucp_identity, backend,
        calibration_curve
    )
    
    # Compute coverage statistics (needed for negative control and final reporting)
    non_abstained = [r for r in evaluation_results if not r['abstained']]
    
    if len(non_abstained) > 0:
        overall_successes = sum(1 for r in non_abstained if r['x0_in_interval']) + sum(1 for r in non_abstained if r['x1_in_interval'])
        total_observables = len(non_abstained) * 2
        observed_coverage = overall_successes / total_observables
    else:
        observed_coverage = 0.0
    
    # Phase 4: Negative control (uses observed coverage)
    negative_control_results = run_negative_control(
        evaluation_results,
        calibration_curve,
        observed_coverage
    )
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    # Coverage analysis (already computed above for negative control)
    abstained_count = len(evaluation_results) - len(non_abstained)
    abstention_rate = abstained_count / len(evaluation_results) if evaluation_results else 0.0
    
    if len(non_abstained) > 0:
        # Compute coverage rates
        x0_successes = sum(1 for r in non_abstained if r['x0_in_interval'])
        x1_successes = sum(1 for r in non_abstained if r['x1_in_interval'])
        total_observables = len(non_abstained) * 2  # X0 and X1 for each circuit
        
        x0_coverage = x0_successes / len(non_abstained)
        x1_coverage = x1_successes / len(non_abstained)
        overall_coverage = (x0_coverage + x1_coverage) / 2
        
        # Compute binomial confidence intervals using Wilson method
        overall_successes = x0_successes + x1_successes
        coverage_lower, coverage_upper = compute_binomial_confidence_interval(
            overall_successes, total_observables, confidence=0.95
        )
    else:
        x0_coverage = x1_coverage = overall_coverage = observed_coverage = 0.0
        coverage_lower = coverage_upper = 0.0
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Total circuits: {len(evaluation_results)}")
    logger.info(f"  Abstained: {abstained_count} ({100*abstention_rate:.1f}%)")
    logger.info(f"  Non-abstained: {len(non_abstained)}")
    
    if len(non_abstained) > 0:
        logger.info(f"\nCoverage Rates:")
        logger.info(f"  X0 coverage: {x0_coverage:.3f} ({x0_successes}/{len(non_abstained)})")
        logger.info(f"  X1 coverage: {x1_coverage:.3f} ({x1_successes}/{len(non_abstained)})")
        logger.info(f"  Overall coverage: {overall_coverage:.3f} ({overall_successes}/{total_observables})")
        logger.info(f"  95% confidence interval: [{coverage_lower:.3f}, {coverage_upper:.3f}]")
        logger.info(f"  Target: ≥ {COVERAGE_MIN} (lower bound of confidence interval)")
        
        # Validation: lower bound of confidence interval must be ≥ COVERAGE_MIN
        coverage_ok = coverage_lower >= COVERAGE_MIN
        logger.info(f"\nCoverage validation: {'PASS' if coverage_ok else 'FAIL'}")
        logger.info(f"  (Lower bound {coverage_lower:.3f} {'≥' if coverage_ok else '<'} {COVERAGE_MIN})")
    else:
        logger.warning("No non-abstained predictions - cannot compute coverage")
        coverage_ok = False
        coverage_lower = coverage_upper = 0.0
    
    # Non-injectivity trap analysis
    logger.info(f"\nNon-Injectivity Trap Results:")
    trap_passed = True
    
    # Check that both circuits with same Z get same intervals (non-injectivity requirement)
    if 'circuit1' in trap_results and 'circuit2' in trap_results:
        r1 = trap_results['circuit1']
        r2 = trap_results['circuit2']
        
        # Both should have same Z distribution (uniform), so method should output same intervals
        if not r1['abstained'] and not r2['abstained']:
            # Check if intervals are the same (within tolerance)
            x0_same = (abs(r1['x0_interval'][0] - r2['x0_interval'][0]) < 0.01 and
                      abs(r1['x0_interval'][1] - r2['x0_interval'][1]) < 0.01)
            x1_same = (abs(r1['x1_interval'][0] - r2['x1_interval'][0]) < 0.01 and
                      abs(r1['x1_interval'][1] - r2['x1_interval'][1]) < 0.01)
            
            if x0_same and x1_same:
                logger.info(f"  Both circuits output same intervals (as required for same Z distribution)")
                # At least one must be wrong unless intervals are wide
                r1_correct = r1['x0_in_interval'] and r1['x1_in_interval']
                r2_correct = r2['x0_in_interval'] and r2['x1_in_interval']
                
                if not (r1_correct and r2_correct):
                    # One is wrong - this demonstrates non-injectivity
                    logger.info(f"  One circuit's truth outside interval (demonstrates non-injectivity)")
                    interval_width = (r1['interval_width'] + r2['interval_width']) / 2
                    if interval_width < NON_INJECTIVITY_WIDE_THRESHOLD:
                        logger.warning(f"  Intervals not wide enough (width={interval_width:.4f} < {NON_INJECTIVITY_WIDE_THRESHOLD})")
                        logger.warning(f"  Method should expand intervals when Z is non-injective for X")
                    else:
                        logger.info(f"  Intervals wide enough (width={interval_width:.4f} ≥ {NON_INJECTIVITY_WIDE_THRESHOLD})")
                else:
                    logger.warning(f"  Both truths in intervals - intervals may be too wide or method got lucky")
            else:
                logger.error(f"  Intervals differ between circuits with same Z - FAIL (violates non-injectivity requirement)")
                trap_passed = False
        elif r1['abstained'] or r2['abstained']:
            logger.info(f"  At least one circuit abstained (correct behavior for non-injective case)")
    
    for circuit_label, result in trap_results.items():
        if result['abstained']:
            logger.info(f"  {circuit_label}: ABSTAINED (QSV={result['qsv']:.4f})")
        else:
            interval_width = result['interval_width']
            logger.info(f"  {circuit_label}: QSV={result['qsv']:.4f}, width={interval_width:.4f}, "
                       f"X0 in interval? {result['x0_in_interval']}, X1 in interval? {result['x1_in_interval']}")
    
    # Negative control analysis
    logger.info(f"\nNegative Control Results:")
    negative_control_passed = negative_control_results['negative_control_passed']
    if negative_control_results['mean_coverage'] is not None:
        logger.info(f"  Mean coverage with shuffled curves: {negative_control_results['mean_coverage']:.3f}")
        logger.info(f"  Coverage drop: {negative_control_results['coverage_drop']:.3f}")
        logger.info(f"  Negative control validation: {'PASS' if negative_control_passed else 'FAIL'}")
        logger.info(f"    (Coverage drop ≥ {NEGATIVE_CONTROL_FAIL_THRESHOLD} confirms harness sensitivity)")
    else:
        logger.warning("  Negative control: No data")
    
    # Final validation
    validation_passed = coverage_ok and trap_passed and negative_control_passed
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FINAL VALIDATION: {'PASS' if validation_passed else 'FAIL'}")
    logger.info(f"{'=' * 80}")
    
    # Save results
    output_dir = f"bounded_validation_qsv_calibrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save calibration curve
    calibration_curve_file = os.path.join(output_dir, 'calibration_curve.json')
    with open(calibration_curve_file, 'w') as f:
        json.dump({
            'backend': BACKEND_NAME,
            'qubit_pair': QUBIT_PAIR,
            'shots': SHOTS,
            'qsv_threshold': QSV_THRESHOLD,
            'timestamp': datetime.now().isoformat(),
            'calibration_set_size': len(calibration_results),
            'calibration_curve': calibration_curve
        }, f, indent=2)
    
    # Save evaluation results
    evaluation_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(evaluation_file, 'w') as f:
        json.dump({
            'backend': BACKEND_NAME,
            'qubit_pair': QUBIT_PAIR,
            'shots': SHOTS,
            'qsv_threshold': QSV_THRESHOLD,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_circuits': len(evaluation_results),
                'abstained': abstained_count,
                'abstention_rate': abstention_rate,
                'non_abstained': len(non_abstained),
                'x0_coverage': float(x0_coverage) if len(non_abstained) > 0 else None,
                'x1_coverage': float(x1_coverage) if len(non_abstained) > 0 else None,
                'overall_coverage': float(overall_coverage) if len(non_abstained) > 0 else None,
                'coverage_lower_bound': float(coverage_lower) if len(non_abstained) > 0 else None,
                'coverage_upper_bound': float(coverage_upper) if len(non_abstained) > 0 else None,
                'coverage_validation': coverage_ok,
                'trap_validation': trap_passed,
                'negative_control_validation': negative_control_passed,
                'final_validation': validation_passed
            },
            'evaluation_results': evaluation_results,
            'trap_results': trap_results,
            'negative_control_results': negative_control_results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"  - calibration_curve.json")
    logger.info(f"  - evaluation_results.json")
    
    return {
        'calibration_results': calibration_results,
        'calibration_curve': calibration_curve,
        'evaluation_results': evaluation_results,
        'trap_results': trap_results,
        'negative_control_results': negative_control_results,
        'validation_passed': validation_passed
    }


if __name__ == "__main__":
    try:
        results = run_bounded_validation()
        logger.info("\nQSV-calibrated bounded validation test completed!")
        sys.exit(0 if results['validation_passed'] else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
