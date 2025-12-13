"""
Noise model implementations for the Quantum Eye framework.

This module provides functions to create different types of quantum noise models
for use with the QuantumEyeAdapter. These noise models are used to simulate the
effects of noise on quantum circuits during testing and validation.
"""

from typing import Dict, Any, Optional
import logging
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error, reset_error

logger = logging.getLogger(__name__)

def get_noise_model_by_type(
    noise_type: str, 
    noise_level: float, 
    metadata: Optional[Dict[str, Any]] = None
) -> NoiseModel:
    """
    Create a noise model based on the specified type and parameters.
    
    Args:
        noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'phase', 
                   'combined', or 'reset_noise')
        noise_level: Strength of the noise (typically 0.0-1.0)
        metadata: Additional metadata for noise model creation (e.g., num_qubits)
        
    Returns:
        Qiskit Aer NoiseModel object
    """
    metadata = metadata or {}
    num_qubits = metadata.get("num_qubits", 1)
    
    # Create an empty noise model
    noise_model = NoiseModel()
    
    # Define minimum set of gates affected by noise
    # In Qiskit 2.0, we want to focus on specific gates rather than all gates
    # to avoid unexpected errors from duplicates
    single_qubit_gates = ["x"]  # Just x gate for single-qubit operations
    two_qubit_gates = ["cx"]    # Just cx gate for two-qubit operations
    
    # Convert noise_level to a reasonable range to avoid errors
    noise_level = max(0.01, min(0.9, noise_level))
    
    if noise_type == "depolarizing":
        noise_model = _create_depolarizing_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    elif noise_type == "amplitude_damping":
        noise_model = _create_amplitude_damping_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    elif noise_type == "phase":
        noise_model = _create_phase_damping_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    elif noise_type == "combined":
        noise_model = _create_combined_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    elif noise_type == "reset_noise":
        noise_model = _create_reset_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    else:
        # Default to depolarizing if type is unknown
        logger.warning(f"Unknown noise type '{noise_type}', defaulting to depolarizing")
        noise_model = _create_depolarizing_noise_model(num_qubits, noise_level, single_qubit_gates, two_qubit_gates)
    
    # Verify that noise model contains operations
    if not noise_model.noise_instructions:
        logger.warning("No errors were added to the noise model. Adding fallback noise.")
        fallback_error = depolarizing_error(max(0.01, noise_level), 1)
        
        for q in range(num_qubits):
            noise_model.add_quantum_error(fallback_error, ["x"], [q])
            noise_model.add_quantum_error(fallback_error, ["measure"], [q])
    
    return noise_model

def _create_depolarizing_noise_model(
    num_qubits: int, 
    noise_level: float, 
    single_qubit_gates: list, 
    two_qubit_gates: list
) -> NoiseModel:
    """
    Create a depolarizing noise model.
    
    Depolarizing noise models random X, Y, Z errors with equal probability.
    
    Args:
        num_qubits: Number of qubits in the system
        noise_level: Noise strength parameter
        single_qubit_gates: List of single-qubit gates to apply noise to
        two_qubit_gates: List of two-qubit gates to apply noise to
        
    Returns:
        NoiseModel with depolarizing noise
    """
    noise_model = NoiseModel()
    
    # Depolarizing noise: random X, Y, Z errors with equal probability
    error_1q = depolarizing_error(noise_level, 1)
    
    # Add to all qubits individually (Qiskit 2.0 style)
    for q in range(num_qubits):
        noise_model.add_quantum_error(error_1q, single_qubit_gates, [q])
        # Also add noise to measurement operations
        noise_model.add_quantum_error(error_1q, ["measure"], [q])
    
    # Two-qubit error for CX gates - typically stronger than single-qubit errors
    error_2q = depolarizing_error(min(0.9, noise_level * 2), 2)
    
    # Add to all possible qubit pairs
    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            if q1 != q2:  # Skip same qubit operations
                noise_model.add_quantum_error(error_2q, two_qubit_gates, [q1, q2])
    
    logger.info(f"Created depolarizing noise model with level {noise_level}")
    return noise_model

def _create_amplitude_damping_noise_model(
    num_qubits: int, 
    noise_level: float, 
    single_qubit_gates: list, 
    two_qubit_gates: list
) -> NoiseModel:
    """
    Create an amplitude damping noise model.
    
    Amplitude damping models energy relaxation (T1 processes).
    
    Args:
        num_qubits: Number of qubits in the system
        noise_level: Noise strength parameter
        single_qubit_gates: List of single-qubit gates to apply noise to
        two_qubit_gates: List of two-qubit gates to apply noise to
        
    Returns:
        NoiseModel with amplitude damping noise
    """
    noise_model = NoiseModel()
    
    # Amplitude damping: models energy relaxation (T1)
    error_1q = amplitude_damping_error(noise_level, 1)
    
    # Add to all qubits individually
    for q in range(num_qubits):
        noise_model.add_quantum_error(error_1q, single_qubit_gates, [q])
        # Also add noise to measurement operations
        noise_model.add_quantum_error(error_1q, ["measure"], [q])
    
    # For two-qubit operations, use a simple depolarizing error
    # This avoids issues with trying to mix different error types
    error_2q = depolarizing_error(min(0.9, noise_level * 1.5), 2)
    
    # Add to all possible qubit pairs
    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            if q1 != q2:  # Skip same qubit operations
                noise_model.add_quantum_error(error_2q, two_qubit_gates, [q1, q2])
    
    logger.info(f"Created amplitude damping noise model with level {noise_level}")
    return noise_model

def _create_phase_damping_noise_model(
    num_qubits: int, 
    noise_level: float, 
    single_qubit_gates: list, 
    two_qubit_gates: list
) -> NoiseModel:
    """
    Create a phase damping noise model.
    
    Phase damping models pure dephasing (T2 processes).
    
    Args:
        num_qubits: Number of qubits in the system
        noise_level: Noise strength parameter
        single_qubit_gates: List of single-qubit gates to apply noise to
        two_qubit_gates: List of two-qubit gates to apply noise to
        
    Returns:
        NoiseModel with phase damping noise
    """
    noise_model = NoiseModel()
    
    # Phase damping: models pure dephasing (T2)
    error_1q = phase_damping_error(noise_level, 1)
    
    # Add to all qubits individually
    for q in range(num_qubits):
        noise_model.add_quantum_error(error_1q, single_qubit_gates, [q])
        # Also add noise to measurement operations
        noise_model.add_quantum_error(error_1q, ["measure"], [q])
    
    # For two-qubit operations, use a simple depolarizing error
    error_2q = depolarizing_error(min(0.9, noise_level * 1.5), 2)
    
    # Add to all possible qubit pairs
    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            if q1 != q2:  # Skip same qubit operations
                noise_model.add_quantum_error(error_2q, two_qubit_gates, [q1, q2])
    
    logger.info(f"Created phase damping noise model with level {noise_level}")
    return noise_model

def _create_combined_noise_model(
    num_qubits: int, 
    noise_level: float, 
    single_qubit_gates: list, 
    two_qubit_gates: list
) -> NoiseModel:
    """
    Create a combined noise model with both amplitude and phase damping.
    
    This provides a more realistic noise model by combining both T1 and T2 processes.
    
    Args:
        num_qubits: Number of qubits in the system
        noise_level: Noise strength parameter
        single_qubit_gates: List of single-qubit gates to apply noise to
        two_qubit_gates: List of two-qubit gates to apply noise to
        
    Returns:
        NoiseModel with combined noise types
    """
    noise_model = NoiseModel()
    
    # Scale levels to keep overall noise comparable
    amp_level = noise_level * 0.5  # Reduce to prevent overflow
    phase_level = noise_level * 0.7
    
    # Create combined error with both amplitude and phase damping
    amp_error = amplitude_damping_error(amp_level, 1)
    phase_error = phase_damping_error(phase_level, 1)
    
    # Add to all qubits individually (one gate at a time to avoid duplicates)
    for q in range(num_qubits):
        noise_model.add_quantum_error(amp_error, ["x"], [q])
        noise_model.add_quantum_error(phase_error, ["x"], [q])
        # Also add noise to measurement operations
        noise_model.add_quantum_error(amp_error, ["measure"], [q])
        noise_model.add_quantum_error(phase_error, ["measure"], [q])
    
    # For two-qubit operations, use a simple depolarizing error
    error_2q = depolarizing_error(min(0.9, noise_level * 1.5), 2)
    
    # Add to all possible qubit pairs
    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            if q1 != q2:  # Skip same qubit operations
                noise_model.add_quantum_error(error_2q, two_qubit_gates, [q1, q2])
    
    logger.info(f"Created combined noise model with level {noise_level}")
    return noise_model

def _create_reset_noise_model(
    num_qubits: int, 
    noise_level: float, 
    single_qubit_gates: list, 
    two_qubit_gates: list
) -> NoiseModel:
    """
    Create a noise model with reset errors.
    
    This model is particularly effective for testing |0⟩ states as it adds
    errors to reset operations.
    
    Args:
        num_qubits: Number of qubits in the system
        noise_level: Noise strength parameter
        single_qubit_gates: List of single-qubit gates to apply noise to
        two_qubit_gates: List of two-qubit gates to apply noise to
        
    Returns:
        NoiseModel with reset errors
    """
    noise_model = NoiseModel()
    
    # Special noise type that affects |0⟩ states through reset operations
    error_1q = depolarizing_error(max(0.05, noise_level * 0.5), 1)
    
    # Add to all qubits individually
    for q in range(num_qubits):
        noise_model.add_quantum_error(error_1q, single_qubit_gates, [q])
        
        # Add reset error (probability of resetting to |1⟩ instead of |0⟩)
        prob_reset_to_wrong_state = min(0.9, noise_level * 1.2)
        reset_err = reset_error(prob_reset_to_wrong_state)
        noise_model.add_quantum_error(reset_err, ["reset"], [q])
        
        # Add error to measurement
        noise_model.add_quantum_error(error_1q, ["measure"], [q])
    
    # Add errors to two-qubit gates
    error_2q = depolarizing_error(min(0.9, noise_level * 1.5), 2)
    
    # Add to all possible qubit pairs
    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            if q1 != q2:  # Skip same qubit operations
                noise_model.add_quantum_error(error_2q, two_qubit_gates, [q1, q2])
    
    logger.info(f"Created reset noise model with level {noise_level}")
    return noise_model

def get_backend_noise_model(backend, noise_level: float = 1.0) -> NoiseModel:
    """
    Create a noise model based on a backend's noise characteristics.
    
    Args:
        backend: The backend to base the noise model on
        noise_level: Scaling factor for noise (1.0 = actual device noise)
        
    Returns:
        NoiseModel based on backend properties
    """
    try:
        # Create noise model from backend properties
        noise_model = NoiseModel.from_backend(backend)
        logger.info(f"Created noise model from backend {backend.name}")
        
        # If noise_level is not 1.0, apply scaling
        # Note: This is a simplified approach - real scaling would be more complex
        if noise_level != 1.0:
            logger.warning(f"Scaled noise model not fully implemented, using original model")
            
        return noise_model
    except Exception as e:
        logger.error(f"Failed to create noise model from backend: {e}")
        # Fall back to a generic depolarizing noise model
        num_qubits = getattr(backend, 'num_qubits', 5)
        return get_noise_model_by_type('depolarizing', noise_level, {'num_qubits': num_qubits})