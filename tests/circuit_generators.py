"""
Parameterized circuit generators for QAT-compliant tests.

Generates circuits that are NOT trivially GHZ-like, with random entangling
patterns and parameterized rotations. Enforces anti-GHZ constraints to prevent
obvious patterns that could be dismissed as "just GHZ again".
"""

import numpy as np
from qiskit import QuantumCircuit
from typing import Optional


def generate_parameterized_circuit(
    n_qubits: int, 
    depth: int, 
    theta: float, 
    seed: Optional[int] = None
) -> QuantumCircuit:
    """
    Generate a parameterized entangling circuit that avoids GHZ-like patterns.
    
    Supports n_qubits up to 10. Anti-GHZ constraints enforced:
    - No single root qubit controls all CXs (max ceil(k/2) per layer)
    - At least one non-neighbor CX per layer when n_qubits >= 5
    - Rotations interleaved on multiple qubits
    
    Args:
        n_qubits: Number of qubits (2-10 supported)
        depth: Circuit depth (number of entangling layers)
        theta: Rotation angle parameter
        seed: Random seed for reproducibility
        
    Returns:
        QuantumCircuit with parameterized entangling structure
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_qubits < 2 or n_qubits > 10:
        raise ValueError(f"n_qubits must be between 2 and 10, got {n_qubits}")
    
    qc = QuantumCircuit(n_qubits)
    
    # Number of CX gates per layer: approximately n_qubits//2
    cx_per_layer = max(1, n_qubits // 2)
    max_controls_per_qubit = (cx_per_layer + 1) // 2  # ceil(k/2)
    
    for layer in range(depth):
        # Interleave rotations on multiple qubits
        num_rotations = min(3, n_qubits)
        rotation_qubits = np.random.choice(n_qubits, size=num_rotations, replace=False)
        for q in rotation_qubits:
            # Use different rotation types to avoid patterns
            if layer % 3 == 0:
                qc.ry(theta + layer * 0.1, q)
            elif layer % 3 == 1:
                qc.rx(theta + layer * 0.1, q)
            else:
                qc.rz(theta + layer * 0.1, q)
        
        # Track CX gates in this layer
        layer_control_counts = {i: 0 for i in range(n_qubits)}
        has_non_neighbor = False
        
        # Add entangling gates
        for _ in range(cx_per_layer):
            available_pairs = []
            for i in range(n_qubits):
                for j in range(n_qubits):
                    if i != j:
                        # Check constraints
                        # 1. Don't exceed max controls per qubit in this layer
                        if layer_control_counts[i] >= max_controls_per_qubit:
                            continue
                        # 2. For n >= 5, ensure at least one non-neighbor
                        if n_qubits >= 5 and not has_non_neighbor and abs(i - j) <= 1:
                            continue  # Skip neighbors until we have a non-neighbor
                        available_pairs.append((i, j))
            
            # If no pairs available (shouldn't happen), fall back to any pair
            if not available_pairs:
                for i in range(n_qubits):
                    for j in range(n_qubits):
                        if i != j and layer_control_counts[i] < max_controls_per_qubit:
                            available_pairs.append((i, j))
            
            if available_pairs:
                # Score pairs: prefer non-neighbors, reverse direction, balanced control
                scored_pairs = []
                for ctrl, tgt in available_pairs:
                    score = 1.0
                    # Prefer non-neighbors (especially if we don't have one yet)
                    if abs(ctrl - tgt) > 1:
                        score *= 2.0
                        if not has_non_neighbor:
                            score *= 3.0  # Strong preference for first non-neighbor
                    # Prefer reverse direction
                    if ctrl > tgt:
                        score *= 1.5
                    # Prefer balanced control distribution
                    if layer_control_counts[ctrl] < layer_control_counts.get(tgt, 0):
                        score *= 1.2
                    scored_pairs.append((score, (ctrl, tgt)))
                
                # Choose pair weighted by score
                scores, pairs = zip(*scored_pairs)
                probs = np.array(scores) / sum(scores)
                chosen_idx = np.random.choice(len(pairs), p=probs)
                control, target = pairs[chosen_idx]
                
                qc.cx(control, target)
                layer_control_counts[control] += 1
                
                # Track if we have a non-neighbor
                if abs(control - target) > 1:
                    has_non_neighbor = True
    
    return qc


def get_recommended_depth(n_qubits: int) -> int:
    """
    Get recommended circuit depth for given number of qubits.
    
    Depth scales with qubit count to avoid benchmarking near-product states.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Recommended depth
    """
    depth_map = {
        2: 2,
        3: 2,
        4: 3,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 5,
        10: 6
    }
    return depth_map.get(n_qubits, max(2, n_qubits // 2))


def generate_basis_rotation_circuit(
    base_circuit: QuantumCircuit, 
    basis: str
) -> QuantumCircuit:
    """
    Generate a circuit that measures in the specified basis.
    
    Args:
        base_circuit: Base quantum circuit (without measurements)
        basis: Measurement basis ('Z', 'X', or 'Y')
        
    Returns:
        QuantumCircuit with basis rotation gates and measurements
    """
    qc = base_circuit.copy()
    n_qubits = qc.num_qubits
    
    if basis == 'Z':
        # Z basis: no rotation needed, just measure
        pass
    elif basis == 'X':
        # X basis: apply H gates before measurement
        for q in range(n_qubits):
            qc.h(q)
    elif basis == 'Y':
        # Y basis: apply S†H gates before measurement
        for q in range(n_qubits):
            qc.sdg(q)
            qc.h(q)
    else:
        raise ValueError(f"Unknown basis: {basis}. Must be 'Z', 'X', or 'Y'")
    
    # Add measurements
    qc.measure_all()
    
    return qc

