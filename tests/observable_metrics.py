"""
Observable accuracy metrics for QAT-compliant tests.

Computes observable errors, expectation values, and comparison metrics.
Includes bitstring conversion helper for consistent eigenvalue calculations.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def bitstring_to_eigenvalues(bitstring: str, pauli_string: str) -> float:
    """
    Convert a bitstring to eigenvalue for a given Pauli string.
    
    Consistent with Qiskit's little-endian measurement ordering:
    - bitstring[0] corresponds to qubit 0 (rightmost in binary representation)
    - For Z: eigenvalue is +1 for |0⟩, -1 for |1⟩
    - For X: eigenvalue is +1 for |+⟩, -1 for |−⟩
    - For Y: eigenvalue is +1 for |+i⟩, -1 for |−i⟩
    
    Args:
        bitstring: Measurement outcome bitstring (e.g., "01")
        pauli_string: Pauli string (e.g., "ZI", "IX", "YY")
        
    Returns:
        Product of eigenvalues for each qubit
    """
    if len(bitstring) != len(pauli_string):
        raise ValueError(f"Bitstring length {len(bitstring)} != Pauli string length {len(pauli_string)}")
    
    eigenvalue = 1.0
    for i, (bit, pauli) in enumerate(zip(bitstring, pauli_string)):
        if pauli == 'I':
            continue
        elif pauli == 'Z':
            # Z: |0⟩ = +1, |1⟩ = -1
            eigenvalue *= 1.0 if bit == '0' else -1.0
        elif pauli == 'X':
            # X: |+⟩ = +1, |−⟩ = -1
            # |+⟩ = (|0⟩ + |1⟩)/√2, |−⟩ = (|0⟩ - |1⟩)/√2
            # For bitstring "0": |0⟩ → |+⟩ with prob 0.5, |−⟩ with prob 0.5
            # For bitstring "1": |1⟩ → |+⟩ with prob 0.5, |−⟩ with prob 0.5
            # We need to compute expectation: ⟨ψ|X|ψ⟩ = prob(|+⟩) - prob(|−⟩)
            # But from Z-basis measurement, we can't directly get X expectation
            # This function is for direct X-basis measurements
            # For X basis measurement: outcome "0" means |+⟩ (+1), "1" means |−⟩ (-1)
            eigenvalue *= 1.0 if bit == '0' else -1.0
        elif pauli == 'Y':
            # Y: |+i⟩ = +1, |−i⟩ = -1
            # For Y basis measurement: outcome "0" means |+i⟩ (+1), "1" means |−i⟩ (-1)
            eigenvalue *= 1.0 if bit == '0' else -1.0
        else:
            raise ValueError(f"Unknown Pauli operator: {pauli}")
    
    return eigenvalue


def compute_expectation_value_from_counts(
    counts: Dict[str, int], 
    pauli_string: str
) -> float:
    """
    Compute expectation value of a Pauli operator from measurement counts.
    
    Args:
        counts: Dictionary of measurement outcomes {bitstring: count}
        pauli_string: Pauli string (e.g., "ZI", "IX", "YY")
        
    Returns:
        Expectation value ⟨Pauli⟩
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    expectation = 0.0
    for bitstring, count in counts.items():
        eigenvalue = bitstring_to_eigenvalues(bitstring, pauli_string)
        probability = count / total
        expectation += eigenvalue * probability
    
    return expectation


def compute_ideal_observables(circuit: QuantumCircuit, include_correlators: bool = True, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Compute ideal expectation values for observables.
    
    Includes single-qubit observables (X_i, Y_i, Z_i) and low-weight correlators
    (Z_i Z_j, X_i X_j) for structure recovery testing.
    
    CRITICAL: Must remove measurements before computing ideal observables.
    
    Uses Statevector methods to compute expectation values directly, ensuring
    consistency with Qiskit's bitstring ordering.
    
    Args:
        circuit: Quantum circuit (may include measurements)
        include_correlators: If True, include 2-qubit correlators (default: True)
        seed: Random seed for selecting correlator pairs (for reproducibility)
        
    Returns:
        Dictionary with keys like "X0", "Y0", "Z0", "Z0Z1", "X0X1", etc.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create clean copy without measurements
    clean_circuit = circuit.copy()
    clean_circuit.remove_final_measurements(inplace=True)
    
    # Get statevector
    sv = Statevector.from_instruction(clean_circuit)
    n_qubits = clean_circuit.num_qubits
    
    observables = {}
    
    # Compute single-qubit observables using Statevector methods
    for q in range(n_qubits):
        # Build Pauli strings
        z_pauli = ''.join(['Z' if i == q else 'I' for i in range(n_qubits)])
        x_pauli = ''.join(['X' if i == q else 'I' for i in range(n_qubits)])
        y_pauli = ''.join(['Y' if i == q else 'I' for i in range(n_qubits)])
        
        # Z observable: measure in Z basis
        z_probs = sv.probabilities_dict()
        z_expectation = 0.0
        for bitstring, prob in z_probs.items():
            eigenvalue = bitstring_to_eigenvalues(bitstring, z_pauli)
            z_expectation += eigenvalue * prob
        observables[f"Z{q}"] = float(z_expectation)
        
        # X observable: apply H gates and measure
        x_circuit = QuantumCircuit(n_qubits)
        x_circuit.h(q)  # Only rotate qubit q
        x_sv = sv.evolve(x_circuit)
        x_probs = x_sv.probabilities_dict()
        x_expectation = 0.0
        for bitstring, prob in x_probs.items():
            eigenvalue = bitstring_to_eigenvalues(bitstring, x_pauli)
            x_expectation += eigenvalue * prob
        observables[f"X{q}"] = float(x_expectation)
        
        # Y observable: apply S†H gates and measure
        y_circuit = QuantumCircuit(n_qubits)
        y_circuit.sdg(q)
        y_circuit.h(q)
        y_sv = sv.evolve(y_circuit)
        y_probs = y_sv.probabilities_dict()
        y_expectation = 0.0
        for bitstring, prob in y_probs.items():
            eigenvalue = bitstring_to_eigenvalues(bitstring, y_pauli)
            y_expectation += eigenvalue * prob
        observables[f"Y{q}"] = float(y_expectation)
    
    # Add 2-qubit correlators for structure recovery
    if include_correlators and n_qubits >= 2:
        # Select 4 fixed pairs (seeded for reproducibility)
        # Generate candidate pairs
        pairs = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                pairs.append((i, j))
        
        # Select up to 4 pairs using deterministic selection
        num_pairs = min(4, len(pairs))
        if seed is not None:
            np.random.seed(seed)
            selected_indices = np.random.choice(len(pairs), size=num_pairs, replace=False)
            selected_pairs = [pairs[idx] for idx in selected_indices]
        else:
            # Default: use first 4 pairs (deterministic)
            selected_pairs = pairs[:num_pairs]
        
        for i, j in selected_pairs:
            # Z_i Z_j correlator
            zz_pauli = ''.join(['Z' if k == i or k == j else 'I' for k in range(n_qubits)])
            zz_probs = sv.probabilities_dict()
            zz_expectation = 0.0
            for bitstring, prob in zz_probs.items():
                eigenvalue = bitstring_to_eigenvalues(bitstring, zz_pauli)
                zz_expectation += eigenvalue * prob
            observables[f"Z{i}Z{j}"] = float(zz_expectation)
            
            # X_i X_j correlator
            xx_circuit = QuantumCircuit(n_qubits)
            xx_circuit.h(i)
            xx_circuit.h(j)
            xx_sv = sv.evolve(xx_circuit)
            xx_pauli = ''.join(['X' if k == i or k == j else 'I' for k in range(n_qubits)])
            xx_probs = xx_sv.probabilities_dict()
            xx_expectation = 0.0
            for bitstring, prob in xx_probs.items():
                eigenvalue = bitstring_to_eigenvalues(bitstring, xx_pauli)
                xx_expectation += eigenvalue * prob
            observables[f"X{i}X{j}"] = float(xx_expectation)
    
    return observables


def compute_observable_error(
    predicted: Dict[str, float], 
    ideal: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute observable error: Δ⟨O⟩ = |⟨O⟩_predicted − ⟨O⟩_ideal|
    
    Args:
        predicted: Dictionary of predicted observables {name: value}
        ideal: Dictionary of ideal observables {name: value}
        
    Returns:
        Dictionary of errors {name: error}
    """
    errors = {}
    all_keys = set(predicted.keys()) | set(ideal.keys())
    
    for key in all_keys:
        pred_val = predicted.get(key, 0.0)
        ideal_val = ideal.get(key, 0.0)
        errors[key] = abs(pred_val - ideal_val)
    
    return errors


def compare_methods(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare results from multiple methods.
    
    Args:
        results: Dictionary {method_name: {observables: {...}, errors: {...}, ...}}
        
    Returns:
        Comparison summary with aggregate metrics
    """
    comparison = {
        'methods': list(results.keys()),
        'observable_errors': {},
        'aggregate_errors': {},
        'basis_counts': {},
        'effective_shots': {}
    }
    
    for method_name, method_results in results.items():
        errors = method_results.get('errors', {})
        comparison['observable_errors'][method_name] = errors
        
        # Aggregate error: mean across all observables
        if errors:
            comparison['aggregate_errors'][method_name] = np.mean(list(errors.values()))
        else:
            comparison['aggregate_errors'][method_name] = float('inf')
        
        # Basis count and effective shots
        comparison['basis_counts'][method_name] = method_results.get('bases_used', 1)
        comparison['effective_shots'][method_name] = method_results.get('effective_shots', 0)
    
    return comparison


def get_scaled_shots(n_qubits: int, base_shots: int = 3000) -> int:
    """
    Get scaled shot count for given number of qubits.
    
    Shot allocation scales with qubit count to keep variance under control:
    S(n) = base_shots * (n/2)
    
    Examples:
        n=2: 3000 shots
        n=4: 6000 shots
        n=6: 9000 shots
        n=8: 12000 shots
        n=10: 15000 shots
    
    Args:
        n_qubits: Number of qubits
        base_shots: Base shot count for 2 qubits (default: 3000)
        
    Returns:
        Scaled shot count
    """
    return int(base_shots * (n_qubits / 2))


def compute_effective_shots(method: str, shots: int, **kwargs) -> int:
    """
    Compute effective shot count accounting for calibration and multi-scale costs.
    
    Args:
        method: Method name ('qe', 'multibasis', 'zne', 'mem')
        shots: Base shot count
        **kwargs: Additional parameters (scales, bases, calibration_shots, etc.)
        
    Returns:
        Effective shot count
    """
    if method == 'qe':
        return shots  # Z-only, no extra cost
    elif method == 'multibasis':
        bases = kwargs.get('bases', 3)
        return shots * bases  # shots per basis × number of bases
    elif method == 'zne':
        scales = kwargs.get('scales', [1, 2, 3])
        bases = kwargs.get('bases', 3)
        return shots * len(scales) * bases  # shots × scales × bases
    elif method == 'mem':
        calibration_shots = kwargs.get('calibration_shots', 0)
        bases = kwargs.get('bases', 3)
        return shots * bases + calibration_shots  # measurement shots + calibration
    else:
        return shots

