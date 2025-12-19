"""
Query ibm_fez backend properties to identify best qubit pairs for hardware validation.

This script connects to ibm_fez, extracts qubit error rates, and ranks qubit pairs
by combined error metrics to help select the optimal pair for experiments.
"""

import sys
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.transpiler import CouplingMap
except ImportError as e:
    print(f"Error: {e}")
    print("Please install qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
    sys.exit(1)


@dataclass
class QubitMetrics:
    """Metrics for a single qubit."""
    qubit: int
    readout_error: float
    t1: Optional[float]
    t2: Optional[float]
    avg_gate_error: float
    combined_score: float  # Lower is better


@dataclass
class QubitPair:
    """Metrics for a qubit pair."""
    q0: int
    q1: int
    readout_error_avg: float
    cx_error: Optional[float]
    t1_avg: Optional[float]
    t2_avg: Optional[float]
    combined_score: float  # Lower is better
    is_connected: bool  # Whether there's a direct CX connection


def get_qubit_metrics(backend, qubit: int) -> QubitMetrics:
    """Extract error metrics for a single qubit."""
    props = backend.properties()
    
    # Readout error
    readout_error = props.readout_error(qubit)
    
    # T1 and T2
    t1 = props.t1(qubit)
    t2 = props.t2(qubit)
    
    # Average single-qubit gate error
    gate_errors = []
    for gate in props.gates:
        if gate.qubits == (qubit,):
            gate_errors.append(gate.parameters[0].value if gate.parameters else 0.0)
    
    avg_gate_error = sum(gate_errors) / len(gate_errors) if gate_errors else 0.0
    
    # Combined score: weighted sum of errors (lower is better)
    # Normalize T1/T2 (longer is better, so invert)
    t1_score = 1.0 / t1 if t1 else 1.0
    t2_score = 1.0 / t2 if t2 else 1.0
    
    combined_score = (
        readout_error * 2.0 +  # Readout is critical
        avg_gate_error * 1.5 +  # Gate errors matter
        t1_score * 0.1 +  # T1 matters less
        t2_score * 0.1   # T2 matters less
    )
    
    return QubitMetrics(
        qubit=qubit,
        readout_error=readout_error,
        t1=t1,
        t2=t2,
        avg_gate_error=avg_gate_error,
        combined_score=combined_score
    )


def get_cx_error(backend, q0: int, q1: int) -> Optional[float]:
    """Get CX gate error between two qubits."""
    props = backend.properties()
    
    for gate in props.gates:
        if gate.gate == 'cx' and set(gate.qubits) == {q0, q1}:
            if gate.parameters:
                return gate.parameters[0].value
    return None


def get_qubit_pair_metrics(backend, q0: int, q1: int, 
                          q0_metrics: QubitMetrics, 
                          q1_metrics: QubitMetrics,
                          coupling_map: CouplingMap) -> QubitPair:
    """Calculate combined metrics for a qubit pair."""
    
    # Average readout error
    readout_error_avg = (q0_metrics.readout_error + q1_metrics.readout_error) / 2.0
    
    # CX error
    cx_error = get_cx_error(backend, q0, q1)
    
    # Average T1 and T2
    t1_avg = (q0_metrics.t1 + q1_metrics.t1) / 2.0 if (q0_metrics.t1 and q1_metrics.t1) else None
    t2_avg = (q0_metrics.t2 + q1_metrics.t2) / 2.0 if (q0_metrics.t2 and q1_metrics.t2) else None
    
    # Check if qubits are directly connected
    is_connected = coupling_map.distance(q0, q1) == 1
    
    # Combined score (lower is better)
    # Prefer connected pairs, lower readout error, lower CX error
    combined_score = (
        readout_error_avg * 2.0 +
        (cx_error * 2.0 if cx_error else 10.0) +  # Heavy penalty if no CX or high error
        (q0_metrics.combined_score + q1_metrics.combined_score) * 0.5
    )
    
    # Bonus for connected pairs
    if not is_connected:
        combined_score += 5.0  # Penalty for unconnected pairs
    
    return QubitPair(
        q0=q0,
        q1=q1,
        readout_error_avg=readout_error_avg,
        cx_error=cx_error,
        t1_avg=t1_avg,
        t2_avg=t2_avg,
        combined_score=combined_score,
        is_connected=is_connected
    )


def get_coupling_map(backend) -> CouplingMap:
    """Get coupling map from backend."""
    if hasattr(backend, 'coupling_map'):
        return CouplingMap(backend.coupling_map)
    elif hasattr(backend, 'configuration'):
        config = backend.configuration()
        if hasattr(config, 'coupling_map'):
            return CouplingMap(config.coupling_map)
    return CouplingMap([])


def main():
    """Query ibm_fez and display best qubit pairs."""
    print("=" * 80)
    print("IBM FEZ QUBIT PROPERTIES QUERY")
    print("=" * 80)
    print()
    
    # Connect to backend
    print("Connecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_fez")
        print(f"[OK] Connected to {backend.name}")
        print(f"  Number of qubits: {backend.num_qubits}")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return
    
    print("\nFetching backend properties...")
    props = backend.properties()
    coupling_map = get_coupling_map(backend)
    
    print(f"[OK] Properties retrieved")
    print(f"  Coupling map: {len(coupling_map.get_edges())} connections")
    
    # Get metrics for all qubits
    print("\nAnalyzing qubit properties...")
    qubit_metrics = {}
    for qubit in range(backend.num_qubits):
        try:
            metrics = get_qubit_metrics(backend, qubit)
            qubit_metrics[qubit] = metrics
        except Exception as e:
            print(f"  Warning: Could not get metrics for qubit {qubit}: {e}")
    
    print(f"[OK] Analyzed {len(qubit_metrics)} qubits")
    
    # Generate all qubit pairs
    print("\nGenerating qubit pair metrics...")
    pairs = []
    for q0 in range(backend.num_qubits):
        for q1 in range(q0 + 1, backend.num_qubits):
            if q0 in qubit_metrics and q1 in qubit_metrics:
                pair = get_qubit_pair_metrics(
                    backend, q0, q1,
                    qubit_metrics[q0],
                    qubit_metrics[q1],
                    coupling_map
                )
                pairs.append(pair)
    
    # Sort by combined score (lower is better)
    pairs.sort(key=lambda p: p.combined_score)
    
    print(f"[OK] Analyzed {len(pairs)} qubit pairs")
    
    # Display results
    print("\n" + "=" * 80)
    print("TOP 10 QUBIT PAIRS (sorted by combined error score, lower is better)")
    print("=" * 80)
    print()
    print(f"{'Rank':<6} {'Qubits':<12} {'Connected':<12} {'Readout Err':<15} {'CX Error':<15} {'T1 (us)':<15} {'T2 (us)':<15} {'Score':<10}")
    print("-" * 100)
    
    for i, pair in enumerate(pairs[:10], 1):
        t1_str = f"{pair.t1_avg:.2f}" if pair.t1_avg else "N/A"
        t2_str = f"{pair.t2_avg:.2f}" if pair.t2_avg else "N/A"
        cx_str = f"{pair.cx_error:.6f}" if pair.cx_error else "N/A"
        
        print(f"{i:<6} {pair.q0}-{pair.q1:<10} {'Yes' if pair.is_connected else 'No':<12} "
              f"{pair.readout_error_avg:<15.6f} {cx_str:<15} {t1_str:<15} {t2_str:<15} {pair.combined_score:<10.4f}")
    
    # Display best connected pair
    connected_pairs = [p for p in pairs if p.is_connected]
    if connected_pairs:
        best_connected = connected_pairs[0]
        print("\n" + "=" * 80)
        print("RECOMMENDED PAIR (best connected qubits)")
        print("=" * 80)
        print(f"Qubits: {best_connected.q0}, {best_connected.q1}")
        print(f"Average readout error: {best_connected.readout_error_avg:.6f}")
        print(f"CX error: {best_connected.cx_error:.6f}" if best_connected.cx_error else "CX error: N/A")
        print(f"Average T1: {best_connected.t1_avg:.2f} us" if best_connected.t1_avg else "Average T1: N/A")
        print(f"Average T2: {best_connected.t2_avg:.2f} us" if best_connected.t2_avg else "Average T2: N/A")
        print(f"Combined score: {best_connected.combined_score:.4f}")
        print()
        print("To use this pair, set initial_layout in transpile:")
        print(f"  initial_layout={{0: {best_connected.q0}, 1: {best_connected.q1}}}")
    
    # Display individual qubit metrics for top pairs
    print("\n" + "=" * 80)
    print("INDIVIDUAL QUBIT METRICS (for top 3 pairs)")
    print("=" * 80)
    for i, pair in enumerate(pairs[:3], 1):
        print(f"\nPair {i}: Qubits {pair.q0}-{pair.q1}")
        for q in [pair.q0, pair.q1]:
            metrics = qubit_metrics[q]
            print(f"  Qubit {q}:")
            print(f"    Readout error: {metrics.readout_error:.6f}")
            print(f"    Avg gate error: {metrics.avg_gate_error:.6f}")
            print(f"    T1: {metrics.t1:.2f} us" if metrics.t1 else f"    T1: N/A")
            print(f"    T2: {metrics.t2:.2f} us" if metrics.t2 else f"    T2: N/A")
            print(f"    Combined score: {metrics.combined_score:.4f}")


if __name__ == "__main__":
    main()

