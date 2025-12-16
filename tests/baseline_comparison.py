"""
Baseline comparison methods for QAT-compliant tests.

All baselines use QuantumEyeAdapter.execute_circuit() to ensure identical noise models.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from qiskit import QuantumCircuit

from tests.circuit_generators import generate_basis_rotation_circuit
from tests.observable_metrics import compute_expectation_value_from_counts


def z_only_baseline(z_counts: Dict[str, int], n_qubits: int) -> Dict[str, float]:
    """
    Z-only baseline: honest prediction that cannot estimate non-commuting observables.
    
    Predicts ⟨Z⟩ from counts, sets ⟨X⟩=⟨Y⟩=0 (mathematically honest - cannot
    predict non-commuting observables from Z-only measurements).
    
    Args:
        z_counts: Z-basis measurement counts {bitstring: count}
        n_qubits: Number of qubits
        
    Returns:
        Dictionary of observables {"X0": 0.0, "Y0": 0.0, "Z0": value, ...}
    """
    observables = {}
    
    # Compute Z observables from counts
    for q in range(n_qubits):
        # Build Pauli string for Z_q
        pauli_string = ''.join(['Z' if i == q else 'I' for i in range(n_qubits)])
        z_expectation = compute_expectation_value_from_counts(z_counts, pauli_string)
        observables[f"Z{q}"] = float(z_expectation)
        
        # Cannot predict X or Y from Z-only measurements
        observables[f"X{q}"] = 0.0
        observables[f"Y{q}"] = 0.0
    
    return observables


def multibasis_observable_estimation(
    circuit: QuantumCircuit,
    adapter,
    shots_per_basis: int,
    noise_type: str,
    noise_level: float,
    include_correlators: bool = True,
    correlator_pairs: Optional[List[Tuple[int, int]]] = None
) -> Tuple[Dict[str, float], int]:
    """
    Multi-Basis baseline: measures Z, X, Y directly and estimates observables.
    
    Uses QuantumEyeAdapter.execute_circuit() for each basis to ensure identical noise.
    
    Args:
        circuit: Base quantum circuit (without measurements)
        adapter: QuantumEyeAdapter instance
        shots_per_basis: Number of shots per basis
        noise_type: Noise type (passed to adapter)
        noise_level: Noise level (passed to adapter)
        include_correlators: If True, compute 2-qubit correlators
        correlator_pairs: List of (i, j) pairs for correlators (if None, auto-select)
        
    Returns:
        Tuple of (observables dictionary, total shots used)
    """
    n_qubits = circuit.num_qubits
    observables = {}
    counts_by_basis = {}
    
    # Measure in each basis
    for basis in ['Z', 'X', 'Y']:
        basis_circuit = generate_basis_rotation_circuit(circuit, basis)
        
        # Execute via adapter (ensures identical noise)
        result = adapter.execute_circuit(
            circuit=basis_circuit,
            shots=shots_per_basis,
            mitigation_enabled=False,  # Baseline without mitigation
            noise_type=noise_type,
            noise_level=noise_level
        )
        
        counts_by_basis[basis] = result['counts']
    
    # Compute single-qubit observables from counts
    for q in range(n_qubits):
        # Z observable from Z-basis measurements
        z_pauli = ''.join(['Z' if i == q else 'I' for i in range(n_qubits)])
        observables[f"Z{q}"] = float(compute_expectation_value_from_counts(
            counts_by_basis['Z'], z_pauli
        ))
        
        # X observable from X-basis measurements
        x_pauli = ''.join(['X' if i == q else 'I' for i in range(n_qubits)])
        observables[f"X{q}"] = float(compute_expectation_value_from_counts(
            counts_by_basis['X'], x_pauli
        ))
        
        # Y observable from Y-basis measurements
        y_pauli = ''.join(['Y' if i == q else 'I' for i in range(n_qubits)])
        observables[f"Y{q}"] = float(compute_expectation_value_from_counts(
            counts_by_basis['Y'], y_pauli
        ))
    
    # Compute 2-qubit correlators if requested
    if include_correlators and n_qubits >= 2:
        if correlator_pairs is None:
            # Auto-select pairs (same as ideal observables)
            pairs = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    pairs.append((i, j))
            correlator_pairs = pairs[:min(4, len(pairs))]
        
        for i, j in correlator_pairs:
            # Z_i Z_j from Z-basis measurements
            zz_pauli = ''.join(['Z' if k == i or k == j else 'I' for k in range(n_qubits)])
            observables[f"Z{i}Z{j}"] = float(compute_expectation_value_from_counts(
                counts_by_basis['Z'], zz_pauli
            ))
            
            # X_i X_j from X-basis measurements
            xx_pauli = ''.join(['X' if k == i or k == j else 'I' for k in range(n_qubits)])
            observables[f"X{i}X{j}"] = float(compute_expectation_value_from_counts(
                counts_by_basis['X'], xx_pauli
            ))
    
    total_shots = 3 * shots_per_basis
    
    return observables, total_shots


def build_mem_calibration_matrix(
    n_qubits: int,
    adapter,
    noise_type: str,
    noise_level: float,
    shots: int
) -> Tuple[np.ndarray, int]:
    """
    Build measurement error mitigation (MEM) calibration matrix.
    
    Prepares |0...0⟩, |0...1⟩, ..., |1...1⟩ and measures in Z basis.
    Uses QuantumEyeAdapter for identical noise.
    
    Args:
        n_qubits: Number of qubits
        adapter: QuantumEyeAdapter instance
        noise_type: Noise type (passed to adapter)
        noise_level: Noise level (passed to adapter)
        shots: Shots per calibration state
        
    Returns:
        Tuple of (calibration matrix, total calibration shots)
    """
    num_states = 2 ** n_qubits
    calibration_matrix = np.zeros((num_states, num_states))
    
    total_calibration_shots = 0
    
    # Prepare each computational basis state
    for prep_state_idx in range(num_states):
        prep_circuit = QuantumCircuit(n_qubits)
        
        # Prepare |prep_state_idx⟩
        prep_bitstring = format(prep_state_idx, f'0{n_qubits}b')
        for q, bit in enumerate(prep_bitstring):
            if bit == '1':
                prep_circuit.x(n_qubits - 1 - q)  # Little-endian
        
        prep_circuit.measure_all()
        
        # Execute via adapter
        result = adapter.execute_circuit(
            circuit=prep_circuit,
            shots=shots,
            mitigation_enabled=False,
            noise_type=noise_type,
            noise_level=noise_level
        )
        
        counts = result['counts']
        total_calibration_shots += shots
        
        # Fill calibration matrix row
        total_counts = sum(counts.values())
        for meas_bitstring, count in counts.items():
            meas_idx = int(meas_bitstring, 2)
            if total_counts > 0:
                calibration_matrix[prep_state_idx, meas_idx] = count / total_counts
    
    return calibration_matrix, total_calibration_shots


def mem_observable_estimation(
    counts_by_basis: Dict[str, Dict[str, int]],
    calibration_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Apply measurement error mitigation to observable estimation.
    
    Args:
        counts_by_basis: Dictionary {basis: {bitstring: count}}
        calibration_matrix: MEM calibration matrix
        
    Returns:
        Dictionary of mitigated observables
    """
    # For simplicity, apply MEM to Z-basis only
    # Full implementation would apply to all bases
    z_counts = counts_by_basis.get('Z', {})
    n_qubits = len(next(iter(z_counts.keys()))) if z_counts else 0
    
    if n_qubits == 0:
        return {}
    
    # Convert counts to probability vector
    num_states = 2 ** n_qubits
    noisy_probs = np.zeros(num_states)
    total = sum(z_counts.values())
    
    for bitstring, count in z_counts.items():
        idx = int(bitstring, 2)
        noisy_probs[idx] = count / total if total > 0 else 0.0
    
    # Apply inverse calibration matrix
    try:
        mitigated_probs = np.linalg.solve(calibration_matrix, noisy_probs)
        mitigated_probs = np.maximum(mitigated_probs, 0.0)  # Ensure non-negative
        mitigated_probs /= mitigated_probs.sum()  # Renormalize
    except np.linalg.LinAlgError:
        # If matrix is singular, use noisy probabilities
        mitigated_probs = noisy_probs
    
    # Convert back to counts and compute observables
    mitigated_counts = {}
    for i, prob in enumerate(mitigated_probs):
        bitstring = format(i, f'0{n_qubits}b')
        mitigated_counts[bitstring] = int(round(prob * total))
    
    observables = {}
    for q in range(n_qubits):
        z_pauli = ''.join(['Z' if i == q else 'I' for i in range(n_qubits)])
        observables[f"Z{q}"] = float(compute_expectation_value_from_counts(
            mitigated_counts, z_pauli
        ))
        # MEM doesn't help with X/Y from Z-only
        observables[f"X{q}"] = 0.0
        observables[f"Y{q}"] = 0.0
    
    return observables


def zne_observable_estimation(
    circuit: QuantumCircuit,
    adapter,
    noise_type: str,
    base_noise_level: float,
    shots_per_basis: int,
    scales: List[int] = [1, 2, 3]
) -> Tuple[Dict[str, float], int]:
    """
    Zero-noise extrapolation (ZNE) baseline.
    
    Runs circuit at multiple noise scales and extrapolates to zero noise.
    Uses QuantumEyeAdapter with scaled noise levels.
    
    Args:
        circuit: Base quantum circuit
        adapter: QuantumEyeAdapter instance
        noise_type: Noise type
        base_noise_level: Base noise level
        shots_per_basis: Shots per basis per scale
        scales: List of noise scale factors
        
    Returns:
        Tuple of (extrapolated observables, effective shots)
    """
    n_qubits = circuit.num_qubits
    observables_by_scale = {}
    
    # Run at each noise scale
    for scale in scales:
        scaled_noise = base_noise_level * scale
        
        # Measure in each basis
        basis_observables = {}
        for basis in ['Z', 'X', 'Y']:
            basis_circuit = generate_basis_rotation_circuit(circuit, basis)
            
            result = adapter.execute_circuit(
                circuit=basis_circuit,
                shots=shots_per_basis,
                mitigation_enabled=False,
                noise_type=noise_type,
                noise_level=scaled_noise
            )
            
            counts = result['counts']
            
            # Compute observables for this basis
            for q in range(n_qubits):
                if basis == 'Z':
                    pauli = ''.join(['Z' if i == q else 'I' for i in range(n_qubits)])
                    key = f"Z{q}"
                elif basis == 'X':
                    pauli = ''.join(['X' if i == q else 'I' for i in range(n_qubits)])
                    key = f"X{q}"
                else:  # Y
                    pauli = ''.join(['Y' if i == q else 'I' for i in range(n_qubits)])
                    key = f"Y{q}"
                
                if key not in basis_observables:
                    basis_observables[key] = []
                
                expectation = compute_expectation_value_from_counts(counts, pauli)
                basis_observables[key].append(expectation)
        
        observables_by_scale[scale] = basis_observables
    
    # Extrapolate to zero noise (linear extrapolation)
    extrapolated_observables = {}
    for obs_name in observables_by_scale[scales[0]].keys():
        values = []
        noise_levels = []
        for scale in scales:
            values.append(observables_by_scale[scale][obs_name][0])
            noise_levels.append(base_noise_level * scale)
        
        # Linear fit: value = a * noise + b, extrapolate to noise=0
        if len(scales) >= 2:
            coeffs = np.polyfit(noise_levels, values, 1)
            extrapolated_value = coeffs[1]  # Intercept (noise=0)
        else:
            extrapolated_value = values[0]
        
        extrapolated_observables[obs_name] = float(extrapolated_value)
    
    effective_shots = shots_per_basis * len(scales) * 3  # scales × bases
    
    return extrapolated_observables, effective_shots

