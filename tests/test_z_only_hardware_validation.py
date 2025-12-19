"""
Hardware Validation: Z-only Prediction vs Direct Measurement on ibm_fez

This experiment tests if Quantum Eye's Z-only predictions match directly 
measured X/Y observables on the same device.

Protocol:
- Two circuits with fixed structure: random rotations → CX → rotations
- Three runs per circuit:
  - Run A: Measure Z only (4096 shots) - Quantum Eye input
  - Run B: Same circuit + H, measure X directly (4096 shots)
  - Run C: Same circuit + S†H, measure Y directly (4096 shots)
- Fixed qubit mapping: {0: 87, 1: 88} on ibm_fez
- Output: Comparison table | Observable | QE(Z-only) | Direct (hardware) | |Δ| | σ |
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
from tests.observable_metrics import compute_expectation_value_from_counts, bitstring_to_eigenvalues

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fixed parameters
BACKEND_NAME = "ibm_fez"
QUBIT_PAIR = {0: 87, 1: 88}  # Fixed mapping from query results
SHOTS = 4096
RANDOM_SEED_CIRCUIT_1 = 42
RANDOM_SEED_CIRCUIT_2 = 43
OPTIMIZATION_LEVEL = 1  # Default


def generate_circuit_angles(seed: int, n_qubits: int = 2) -> List[float]:
    """Generate fixed rotation angles for a circuit using a seed."""
    rng = np.random.RandomState(seed)
    # Generate 8 angles: 4 per qubit (2 rotations per qubit, 2 layers)
    angles = []
    for _ in range(2):  # Two layers
        for _ in range(n_qubits):  # Per qubit
            angles.append(rng.uniform(0, 2 * np.pi))  # RY angle
            angles.append(rng.uniform(0, 2 * np.pi))  # RZ angle
    return angles


def create_circuit(angles: List[float], basis: str = 'Z') -> QuantumCircuit:
    """
    Create a 2-qubit circuit with fixed structure.
    
    Structure: random rotations → CX → rotations → measurement
    
    Args:
        angles: List of 8 angles [ry0_1, rz0_1, ry1_1, rz1_1, ry0_2, rz0_2, ry1_2, rz1_2]
        basis: Measurement basis ('Z', 'X', or 'Y')
    """
    qc = QuantumCircuit(2, 2)
    
    # Layer 1: Random rotations
    qc.ry(angles[0], 0)  # RY on qubit 0
    qc.rz(angles[1], 0)  # RZ on qubit 0
    qc.ry(angles[2], 1)  # RY on qubit 1
    qc.rz(angles[3], 1)  # RZ on qubit 1
    
    # CX gate
    qc.cx(0, 1)
    
    # Layer 2: More rotations
    qc.ry(angles[4], 0)  # RY on qubit 0
    qc.rz(angles[5], 0)  # RZ on qubit 0
    qc.ry(angles[6], 1)  # RY on qubit 1
    qc.rz(angles[7], 1)  # RZ on qubit 1
    
    # Basis change for measurement
    if basis == 'X':
        qc.h(0)
        qc.h(1)
    elif basis == 'Y':
        qc.sdg(0)
        qc.h(0)
        qc.sdg(1)
        qc.h(1)
    # For 'Z', no basis change needed
    
    qc.measure([0, 1], [0, 1])
    return qc


def compute_observable_from_state(state: np.ndarray, pauli_string: str) -> float:
    """Compute expectation value of a Pauli operator from a statevector."""
    sv = Statevector(state)
    pauli_op = SparsePauliOp(pauli_string)
    expectation = sv.expectation_value(pauli_op)
    return float(np.real(expectation))


def compute_binomial_error(p: float, n: int) -> float:
    """Compute binomial standard error: σ = √(p(1-p)/n)"""
    if n == 0:
        return 0.0
    return np.sqrt(p * (1 - p) / n)


def compute_observable_from_counts(counts: Dict[str, int], pauli_string: str) -> Tuple[float, float]:
    """
    Compute expectation value and uncertainty from measurement counts.
    
    Returns:
        (expectation_value, uncertainty)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    
    # For single-qubit observables, compute from probabilities
    if pauli_string == 'ZI':
        # Z on qubit 0
        p0 = sum(count for bitstring, count in counts.items() if bitstring[0] == '0') / total
        expectation = 2 * p0 - 1  # +1 for |0⟩, -1 for |1⟩
        uncertainty = compute_binomial_error(p0, total) * 2  # Scale by 2 for ±1 eigenvalues
    elif pauli_string == 'IZ':
        # Z on qubit 1
        p0 = sum(count for bitstring, count in counts.items() if bitstring[1] == '0') / total
        expectation = 2 * p0 - 1
        uncertainty = compute_binomial_error(p0, total) * 2
    elif pauli_string == 'XI':
        # X on qubit 0 (from X-basis measurement)
        p_plus = sum(count for bitstring, count in counts.items() if bitstring[0] == '0') / total
        expectation = 2 * p_plus - 1  # +1 for |+⟩, -1 for |−⟩
        uncertainty = compute_binomial_error(p_plus, total) * 2
    elif pauli_string == 'IX':
        # X on qubit 1
        p_plus = sum(count for bitstring, count in counts.items() if bitstring[1] == '0') / total
        expectation = 2 * p_plus - 1
        uncertainty = compute_binomial_error(p_plus, total) * 2
    elif pauli_string == 'YI':
        # Y on qubit 0 (from Y-basis measurement)
        p_plus_i = sum(count for bitstring, count in counts.items() if bitstring[0] == '0') / total
        expectation = 2 * p_plus_i - 1  # +1 for |+i⟩, -1 for |−i⟩
        uncertainty = compute_binomial_error(p_plus_i, total) * 2
    elif pauli_string == 'IY':
        # Y on qubit 1
        p_plus_i = sum(count for bitstring, count in counts.items() if bitstring[1] == '0') / total
        expectation = 2 * p_plus_i - 1
        uncertainty = compute_binomial_error(p_plus_i, total) * 2
    else:
        # General case: use eigenvalue method
        expectation = compute_expectation_value_from_counts(counts, pauli_string)
        # For general case, estimate uncertainty from variance
        # This is approximate but reasonable for comparison
        variance = sum(
            (bitstring_to_eigenvalues(bitstring, pauli_string) - expectation) ** 2 * (count / total)
            for bitstring, count in counts.items()
        )
        uncertainty = np.sqrt(variance / total)
    
    return expectation, uncertainty


def run_experiment():
    """Run the complete hardware validation experiment."""
    logger.info("=" * 80)
    logger.info("HARDWARE VALIDATION: Z-only Prediction vs Direct Measurement")
    logger.info("=" * 80)
    logger.info(f"Backend: {BACKEND_NAME}")
    logger.info(f"Qubit mapping: {QUBIT_PAIR}")
    logger.info(f"Shots per run: {SHOTS}")
    logger.info(f"Circuit 1 seed: {RANDOM_SEED_CIRCUIT_1}")
    logger.info(f"Circuit 2 seed: {RANDOM_SEED_CIRCUIT_2}")
    logger.info("=" * 80)
    
    # Initialize adapter
    logger.info("\nInitializing adapter...")
    adapter = QuantumEyeAdapter({
        'backend_type': 'real',
        'backend_name': BACKEND_NAME,
        'default_shots': SHOTS,
        'optimization_level': OPTIMIZATION_LEVEL
    })
    
    # Get backend for transpilation
    backend = adapter.backend
    
    # Generate fixed angles for both circuits
    angles_1 = generate_circuit_angles(RANDOM_SEED_CIRCUIT_1)
    angles_2 = generate_circuit_angles(RANDOM_SEED_CIRCUIT_2)
    
    logger.info(f"\nCircuit 1 angles: {[f'{a:.4f}' for a in angles_1]}")
    logger.info(f"Circuit 2 angles: {[f'{a:.4f}' for a in angles_2]}")
    
    # Phase 1: Collect all circuits first
    logger.info(f"\n{'=' * 80}")
    logger.info("PHASE 1: Preparing all circuits for batch execution")
    logger.info(f"{'=' * 80}")
    
    all_circuits = []
    circuit_metadata = []
    from qiskit.transpiler import Layout
    
    for circuit_num, angles, seed in [(1, angles_1, RANDOM_SEED_CIRCUIT_1), 
                                      (2, angles_2, RANDOM_SEED_CIRCUIT_2)]:
        logger.info(f"\nPreparing circuits for circuit {circuit_num}...")
        
        for basis, run_type in [('Z', 'Z-only'), ('X', 'X-direct'), ('Y', 'Y-direct')]:
            circuit = create_circuit(angles, basis=basis)
            
            # Set fixed layout
            layout_dict = {circuit.qubits[i]: QUBIT_PAIR[i] for i in range(2)}
            layout = Layout(layout_dict)
            
            logger.info(f"  Transpiling {basis}-basis circuit {circuit_num} with fixed layout {QUBIT_PAIR}...")
            transpiled_circuit = transpile(
                circuit,
                backend=backend,
                initial_layout=layout,
                optimization_level=OPTIMIZATION_LEVEL
            )
            
            all_circuits.append(transpiled_circuit)
            circuit_metadata.append({
                'circuit_num': circuit_num,
                'angles': angles,
                'seed': seed,
                'basis': basis,
                'run_type': run_type,
                'needs_mitigation': (basis == 'Z')
            })
    
    logger.info(f"\nPrepared {len(all_circuits)} circuits for batch execution")
    
    # Phase 2: Submit all circuits in a single batch
    logger.info(f"\n{'=' * 80}")
    logger.info("PHASE 2: Submitting all circuits in batch mode")
    logger.info(f"{'=' * 80}")
    
    from qiskit_ibm_runtime import Batch, SamplerV2
    
    with Batch(backend=backend) as batch:
        sampler_options = {
            "default_shots": SHOTS,
            "twirling": {"enable_measure": True}
        }
        sampler = SamplerV2(mode=batch, options=sampler_options)
        
        logger.info(f"Submitting batch job with {len(all_circuits)} circuits...")
        job = sampler.run(all_circuits)
        
        logger.info(f"Batch job {job.job_id()} submitted, waiting for completion...")
        adapter._wait_for_job(job)
        
        # Get all results
        batch_result = job.result()
        logger.info(f"Batch job completed successfully with {len(batch_result)} results")
    
    # Phase 3: Process results
    logger.info(f"\n{'=' * 80}")
    logger.info("PHASE 3: Processing batch results")
    logger.info(f"{'=' * 80}")
    
    results = []
    
    # Group results by circuit number
    circuit_results = {}
    for idx, pub_result in enumerate(batch_result):
        meta = circuit_metadata[idx]
        circuit_num = meta['circuit_num']
        
        # Extract counts from result
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
        
        if circuit_num not in circuit_results:
            circuit_results[circuit_num] = {}
        circuit_results[circuit_num][meta['basis']] = {
            'counts': counts,
            'meta': meta
        }
    
    # Process each circuit's results
    for circuit_num in sorted(circuit_results.keys()):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROCESSING CIRCUIT {circuit_num}")
        logger.info(f"{'=' * 80}")
        
        circ_data = circuit_results[circuit_num]
        meta_z = circ_data['Z']['meta']
        angles = meta_z['angles']
        seed = meta_z['seed']
        
        # Get Z-basis counts and apply mitigation
        logger.info(f"\nProcessing Z-basis results (Quantum Eye input)...")
        z_counts = circ_data['Z']['counts']
        logger.info(f"Z-basis counts: {z_counts}")
        
        # Apply mitigation to Z-basis counts
        # Create a dummy 2-qubit circuit for mitigation (it only uses counts anyway)
        dummy_circuit = QuantumCircuit(2)
        mitigation_result = adapter._apply_mitigation_to_counts(
            counts=z_counts,
            circuit=dummy_circuit,
            reference_label=None,
            component_weights=None
        )
        
        # Extract mitigated state from result
        mitigated_state = mitigation_result.get('mitigated_state')
        
        # Log mitigation result for debugging
        logger.info(f"Mitigation result keys: {list(mitigation_result.keys())}")
        logger.info(f"Mitigation method: {mitigation_result.get('mitigation_method', 'unknown')}")
        logger.info(f"Mitigation successful: {mitigation_result.get('mitigated', False)}")
        
        if mitigated_state is None:
            logger.error("Failed to get mitigated state from Quantum Eye")
            # Fallback: use raw counts to reconstruct state (approximate)
            total = sum(z_counts.values())
            mitigated_state = np.zeros(4, dtype=complex)
            for bitstring, count in z_counts.items():
                idx = int(bitstring, 2)
                mitigated_state[idx] = np.sqrt(count / total)
            mitigated_state = mitigated_state / np.linalg.norm(mitigated_state)
            logger.warning("Using approximate state reconstruction from counts")
        
        # Compute X and Y observables from Quantum Eye state
        logger.info("Computing X and Y observables from Quantum Eye state...")
        qe_x0 = compute_observable_from_state(mitigated_state, 'XI')
        qe_x1 = compute_observable_from_state(mitigated_state, 'IX')
        qe_y0 = compute_observable_from_state(mitigated_state, 'YI')
        qe_y1 = compute_observable_from_state(mitigated_state, 'IY')
        
        # Estimate uncertainty using formula: σ = √(max(0, 1 - val²) / shots)
        qe_x0_unc = np.sqrt(max(0.0, 1.0 - qe_x0**2) / SHOTS)
        qe_x1_unc = np.sqrt(max(0.0, 1.0 - qe_x1**2) / SHOTS)
        qe_y0_unc = np.sqrt(max(0.0, 1.0 - qe_y0**2) / SHOTS)
        qe_y1_unc = np.sqrt(max(0.0, 1.0 - qe_y1**2) / SHOTS)
        
        logger.info(f"QE predictions: X0={qe_x0:.4f}, X1={qe_x1:.4f}, Y0={qe_y0:.4f}, Y1={qe_y1:.4f}")
        
        # Process X-basis direct measurement
        logger.info(f"\nProcessing X-basis direct measurement...")
        x_counts = circ_data['X']['counts']
        logger.info(f"X-basis counts: {x_counts}")
        
        x0_direct, x0_unc = compute_observable_from_counts(x_counts, 'XI')
        x1_direct, x1_unc = compute_observable_from_counts(x_counts, 'IX')
        
        logger.info(f"Direct X measurements: X0={x0_direct:.4f}±{x0_unc:.4f}, X1={x1_direct:.4f}±{x1_unc:.4f}")
        
        # Process Y-basis direct measurement
        logger.info(f"\nProcessing Y-basis direct measurement...")
        y_counts = circ_data['Y']['counts']
        logger.info(f"Y-basis counts: {y_counts}")
        
        y0_direct, y0_unc = compute_observable_from_counts(y_counts, 'YI')
        y1_direct, y1_unc = compute_observable_from_counts(y_counts, 'IY')
        
        logger.info(f"Direct Y measurements: Y0={y0_direct:.4f}±{y0_unc:.4f}, Y1={y1_direct:.4f}±{y1_unc:.4f}")
        
        # Store results
        results.append({
            'circuit': circuit_num,
            'angles': [float(a) for a in angles],
            'seed': seed,
            'qe': {
                'X0': {'value': qe_x0, 'uncertainty': qe_x0_unc},
                'X1': {'value': qe_x1, 'uncertainty': qe_x1_unc},
                'Y0': {'value': qe_y0, 'uncertainty': qe_y0_unc},
                'Y1': {'value': qe_y1, 'uncertainty': qe_y1_unc}
            },
            'direct': {
                'X0': {'value': x0_direct, 'uncertainty': x0_unc},
                'X1': {'value': x1_direct, 'uncertainty': x1_unc},
                'Y0': {'value': y0_direct, 'uncertainty': y0_unc},
                'Y1': {'value': y1_direct, 'uncertainty': y1_unc}
            },
            'counts': {
                'Z': z_counts,
                'X': x_counts,
                'Y': y_counts
            }
        })
    
    # Generate comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'=' * 80}\n")
    
    print("| Observable | QE(Z-only) | Direct (hardware) | |Delta| | sigma |")
    print("|------------|------------|-------------------|-----|-----|")
    
    table_data = []
    
    for result in results:
        circuit_num = result['circuit']
        
        for obs in ['X0', 'X1', 'Y0', 'Y1']:
            qe_val = result['qe'][obs]['value']
            qe_unc = result['qe'][obs]['uncertainty']
            direct_val = result['direct'][obs]['value']
            direct_unc = result['direct'][obs]['uncertainty']
            
            delta = abs(qe_val - direct_val)
            # Use the larger uncertainty as the reference σ
            sigma = max(qe_unc, direct_unc)
            
            table_data.append({
                'observable': f"⟨{obs}⟩ (circuit {circuit_num})",
                'qe_value': qe_val,
                'qe_unc': qe_unc,
                'direct_value': direct_val,
                'direct_unc': direct_unc,
                'delta': delta,
                'sigma': sigma
            })
            
            obs_symbol = obs.replace('X', 'X').replace('Y', 'Y').replace('0', '0').replace('1', '1')
            print(f"| <{obs_symbol}> (circuit {circuit_num}) | {qe_val:.4f} +/- {qe_unc:.4f} | {direct_val:.4f} +/- {direct_unc:.4f} | {delta:.4f} | {sigma:.4f} |")
    
    # Save results
    output_dir = f"z_only_hardware_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'backend': BACKEND_NAME,
            'qubit_pair': QUBIT_PAIR,
            'shots': SHOTS,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'table_data': table_data
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}/")
    
    # Check if predictions are within 2-3σ
    logger.info(f"\n{'=' * 80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")
    
    within_2sigma = 0
    within_3sigma = 0
    total = len(table_data)
    
    for entry in table_data:
        if entry['delta'] <= 2 * entry['sigma']:
            within_2sigma += 1
        if entry['delta'] <= 3 * entry['sigma']:
            within_3sigma += 1
    
    logger.info(f"Predictions within 2σ: {within_2sigma}/{total} ({100*within_2sigma/total:.1f}%)")
    logger.info(f"Predictions within 3σ: {within_3sigma}/{total} ({100*within_3sigma/total:.1f}%)")
    
    return results, table_data


if __name__ == "__main__":
    try:
        results, table_data = run_experiment()
        logger.info("\nExperiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)

