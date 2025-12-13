"""
Centralized imports for the Quantum Eye framework.

This module provides centralized imports for the entire Quantum Eye framework,
helping to prevent circular imports and make the codebase more maintainable.
"""

# Standard library imports
import logging
from typing import Dict, Optional, Any, Union, List, Tuple

# NumPy and SciPy
import numpy as np
from scipy import linalg

# Qiskit Core
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Providers
from qiskit.providers.fake_provider import GenericBackendV2

# Qiskit Aer
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.noise import NoiseModel

# Conditional IBM Runtime imports
try:
    # Attempt to import specific fake backends from qiskit-ibm-runtime if available
    from qiskit_ibm_runtime.fake_provider import (
        FakeManilaV2, FakeLagosV2, FakeBelemV2, FakeQuitoV2,
        FakeMontrealV2, FakeMumbaiV2, FakeCairoV2,
        FakeWashingtonV2, 
    )
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False
    # Define empty placeholders for IDE type checking
    FakeManilaV2 = None
    FakeLagosV2 = None
    FakeBelemV2 = None
    FakeQuitoV2 = None
    FakeMontrealV2 = None
    FakeMumbaiV2 = None
    FakeCairoV2 = None
    FakeWashingtonV2 = None
    
# Configure logging
logger = logging.getLogger(__name__)