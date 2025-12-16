"""
QuantumEyeAdapter for the Quantum Eye Framework

This module provides a unified adapter for integrating the Quantum Eye framework
with Qiskit. It supports both simulated backends (for testing and validation)
and real IBM Quantum hardware for actual quantum execution.
"""

from typing import Dict, Optional, Any, Union, List, Tuple
import numpy as np
import logging
import time
from scipy import linalg
import scipy.sparse as sparse

# Import Qiskit components
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers import Backend
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JobStatus

# Import Qiskit fake provider and Aer
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.noise import NoiseModel

# Import noise models
try:
    from .noise_models import get_noise_model_by_type
except ImportError:
    # Fallback for missing noise_models module
    def get_noise_model_by_type(noise_type, noise_level, options=None):
        """Simple fallback noise model"""
        noise_model = NoiseModel()
        return noise_model

# Direct imports for IBM Quantum Runtime - FORCE THEM TO BE AVAILABLE
HAS_IBM_RUNTIME = False
try:
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Batch, SamplerV2, EstimatorV2, Options
    from qiskit_ibm_runtime.options import EstimatorOptions
    from qiskit_ibm_runtime.exceptions import (RuntimeJobFailureError, 
                                            RuntimeJobTimeoutError,
                                            RuntimeInvalidStateError)
    HAS_IBM_RUNTIME = True
    # Import fake IBM backends
    try:
        from qiskit_ibm_runtime.fake_provider import (
            FakeManilaV2, FakeLagosV2, FakeBelemV2, FakeQuitoV2,
            FakeMontrealV2, FakeMumbaiV2, FakeCairoV2,
            FakeWashingtonV2, FakeTorinoV2
        )
        print(f"IBM Quantum Runtime fake backends imported successfully")
    except ImportError as e:
        print(f"Failed to import fake backends: {str(e)}")
    
    print(f"IBM Quantum Runtime available (version: {qiskit_ibm_runtime.__version__})")
except ImportError as e:
    HAS_IBM_RUNTIME = False
    print(f"IBM Quantum Runtime not available: {str(e)}")

# Configure logging
logger = logging.getLogger(__name__)

class QuantumEyeAdapter:
    """
    Unified adapter for the Quantum Eye framework that supports both
    simulated backends and real IBM Quantum hardware.
    
    This adapter provides methods for executing circuits, applying error
    mitigation, and analyzing results using the Quantum Eye framework.
    """
    
    # Available fake backends in Qiskit 2.0 that work well with this adapter
    AVAILABLE_FAKE_BACKENDS = {
        # 5-7 qubit devices (good for basic testing)
        'manila': {'n_qubits': 5, 'coupling_map': [[0, 1], [1, 2], [2, 3], [3, 4]]},
        'lagos': {'n_qubits': 7, 'coupling_map': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]},
        'belem': {'n_qubits': 5, 'coupling_map': [[0, 1], [1, 2], [2, 3], [3, 4]]},
        'quito': {'n_qubits': 5, 'coupling_map': [[0, 1], [1, 2], [2, 3], [3, 4]]},
        
        # Medium sized devices (good for more complex circuit testing)
        'montreal': {'n_qubits': 27, 'coupling_map': 'heavy_hex'},
        'mumbai': {'n_qubits': 27, 'coupling_map': 'heavy_hex'},
        'cairo': {'n_qubits': 27, 'coupling_map': 'heavy_hex'},
        
        # Large devices (for scaling tests)
        'washington': {'n_qubits': 127, 'coupling_map': 'heavy_hex'},
        'kyoto': {'n_qubits': 27, 'coupling_map': 'heavy_hex'},
        'torino': {'n_qubits': 133, 'coupling_map': 'heavy_hex'},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, quantum_eye=None):
        """
        Initialize the QuantumEyeAdapter.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - backend_name: Name of the backend to use (default: 'ibm_simulator')
                - backend_type: 'simulator', 'fake', or 'real' (default: 'simulator')
                - noise_type: Type of noise model to use (default: 'depolarizing')
                - noise_level: Noise level for simulations (default: 0.1)
                - default_shots: Default number of shots for execution (default: 1024)
                - reference_match_threshold: Threshold for reference state matching (default: 0.6)
                - token: IBM Quantum API token (for real hardware)
                - instance: IBM Quantum instance to use (for real hardware)
                - channel: IBM Quantum channel (for real hardware)
            quantum_eye: Optional QuantumEye instance for dependency injection
        """
        logger.info(f"Adapter config received: {config}")
        backend_name = config.get('backend_name', 'aer_simulator')
        logger.info(f"Requested backend: {backend_name}")
        
        self.config = config or {}
        
        # Use injected QuantumEye or create new instance
        if quantum_eye is not None:
            self.quantum_eye = quantum_eye
        else:
            from quantum_eye import QuantumEye
            self.quantum_eye = QuantumEye(self.config)
        
        # Simple dictionary-based reference state registry
        self.reference_states = {}
        
        # Cache for noise models to avoid regenerating them
        self._noise_model_cache = {}
        
        # Error map for backend calibration
        self._error_map = None
        
        # Initialize service and backend
        self._initialize_backend_and_service()
        
        logger.info(f"QuantumEyeAdapter initialized with backend: {self.backend.name}")
    
    def _initialize_backend_and_service(self):
        """Initialize the backend and service based on configuration."""
        backend_name = self.config.get('backend_name', 'aer_simulator')
        backend_type = self.config.get('backend_type', 'simulator')
        
        logger.info(f"Initializing backend with type: '{backend_type}'")
        logger.info(f"Requested backend: '{backend_name}'")
        
        # IBM/simulator logic only
        if backend_type == 'real' and HAS_IBM_RUNTIME:
            self._initialize_real_backend()
        elif backend_type == 'fake' or (backend_type == 'real' and not HAS_IBM_RUNTIME):
            self._initialize_fake_backend()
        else:
            self._initialize_simulator_backend()
    
    def _initialize_real_backend(self):
        """Initialize a real IBM Quantum backend with authentication."""
        backend_name = self.config.get('backend_name', 'ibmq_qasm_simulator')
        
        try:
            logger.info("Attempting to initialize real backend...")
            
            # Check for authentication parameters in config
            token = self.config.get('token', None)
            instance = self.config.get('instance', None)
            channel = self.config.get('channel', 'ibm_cloud')
            
            # Initialize service
            if token:
                # Initialize with provided credentials
                logger.info(f"Initializing QiskitRuntimeService with provided credentials (channel: {channel})")
                self.service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
            else:
                # Try to load saved account
                logger.info("Attempting to initialize QiskitRuntimeService with saved account")
                self.service = QiskitRuntimeService()
            
            logger.info("QiskitRuntimeService initialized successfully")
            
            # Get available backends for verification
            logger.info("Fetching available backends")
            available_backends = self.service.backends()
            available_names = [b.name for b in available_backends]
            logger.info(f"Available backends: {available_names}")
            
            # Verify and get the backend
            if backend_name in available_names:
                self.backend = self.service.backend(backend_name)
                self.backend_type = 'real'
                logger.info(f"Connected to real backend: {self.backend.name}")
            else:
                logger.warning(f"Backend '{backend_name}' not found or not available")
                # Try to use a default backend if available
                if available_names:
                    fallback_name = available_names[0]
                    logger.info(f"Attempting to use '{fallback_name}' as fallback")
                    self.backend = self.service.backend(fallback_name)
                    self.backend_type = 'real'
                else:
                    raise ValueError(f"Backend '{backend_name}' not found and no fallbacks available")
                
            # Store service for use in batch executions
            self._ibm_service = self.service
            
            # Store backend properties for reference
            self.backend_info = {
                'name': self.backend.name,
                'num_qubits': self.backend.num_qubits,
                'basis_gates': self.backend.operation_names,
                'coupling_map': self._get_coupling_map(),
                'simulator': False,
                'local': False,
                'backend_version': getattr(self.backend, 'backend_version', 'unknown')
            }
            
        except Exception as e:
            error_msg = f"Failed to initialize IBM Quantum service or backend: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _initialize_fake_backend(self):
        """Initialize a fake backend that mimics real quantum hardware."""
        backend_name = self.config.get('backend_name', 'manila').lower()
        
        # Check if the backend name is in our list of known backends
        if backend_info := self.AVAILABLE_FAKE_BACKENDS.get(backend_name):
            num_qubits = backend_info['n_qubits']
            coupling_map = backend_info['coupling_map']
            
            # Try to use specific fake backend if IBM Runtime is available
            if HAS_IBM_RUNTIME:
                class_name = f"Fake{backend_name.capitalize()}V2"
                # Look for the backend class in qiskit_ibm_runtime.fake_provider
                try:
                    # Import dynamically to ensure we can access the classes
                    from qiskit_ibm_runtime.fake_provider import (
                        FakeManilaV2, FakeLagosV2, FakeBelemV2, FakeQuitoV2,
                        FakeMontrealV2, FakeMumbaiV2, FakeCairoV2,
                        FakeWashingtonV2, 
                    )
                    
                    # Map of class names to actual classes
                    fake_backend_classes = {
                        "FakeManilaV2": FakeManilaV2,
                        "FakeLagosV2": FakeLagosV2,
                        "FakeBelemV2": FakeBelemV2,
                        "FakeQuitoV2": FakeQuitoV2,
                        "FakeMontrealV2": FakeMontrealV2,
                        "FakeMumbaiV2": FakeMumbaiV2,
                        "FakeCairoV2": FakeCairoV2,
                        "FakeWashingtonV2": FakeWashingtonV2,
                    }
                    
                    # Try to get the class and instantiate it
                    if class_name in fake_backend_classes:
                        backend_class = fake_backend_classes[class_name]
                        self.backend = backend_class()
                        logger.info(f"Using specific fake backend: {class_name}")
                    else:
                        logger.warning(f"Specific fake backend {class_name} not available, using generic backend")
                        self.backend = self._create_generic_backend(num_qubits, coupling_map, backend_name)
                except Exception as e:
                    logger.warning(f"Failed to create specific fake backend: {str(e)}")
                    logger.info("Falling back to generic backend")
                    self.backend = self._create_generic_backend(num_qubits, coupling_map, backend_name)
            else:
                # Use GenericBackendV2 as fallback if IBM Runtime isn't available
                logger.warning("IBM Quantum Runtime not available, using generic backend")
                self.backend = self._create_generic_backend(num_qubits, coupling_map, backend_name)
        else:
            # If backend name is not recognized, log warning and use generic 5-qubit backend
            logger.warning(f"Unknown backend '{backend_name}', defaulting to generic 5-qubit backend")
            self.backend = self._create_generic_backend(5, [[0, 1], [1, 2], [2, 3], [3, 4]], "generic")
        
        self.backend_type = 'fake'
        
        # Store backend characteristics for reference
        self.backend_info = {
            'name': self.backend.name,
            'num_qubits': self.backend.num_qubits,
            'basis_gates': self.backend.operation_names,
            'coupling_map': self._get_coupling_map(),
            'simulator': True,
            'local': True,
            'backend_version': getattr(self.backend, 'backend_version', 'unknown')
        }
        
        logger.info(f"Initialized fake backend: {self.backend.name} with {self.backend.num_qubits} qubits")
    
    def _create_generic_backend(self, num_qubits, coupling_map, name):
        """
        Create a GenericBackendV2 instance with the specified characteristics.
        
        Args:
            num_qubits: Number of qubits for the backend
            coupling_map: Coupling map specification
            name: Name to assign to the backend
            
        Returns:
            GenericBackendV2 instance
        """
        # Handle 'heavy_hex' coupling map
        if coupling_map == 'heavy_hex':
            # Create a simplified heavy-hex topology
            # This is just an approximation of the heavy-hex topology
            coupling_map = []
            for i in range(num_qubits - 1):
                coupling_map.append([i, i+1])
                # Add some additional connections for the hex structure
                if i % 6 == 0 and i+6 < num_qubits:
                    coupling_map.append([i, i+6])
        
        # Create and return the generic backend
        backend = GenericBackendV2(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'],
            seed=42
        )
        
        # Set a custom name if possible
        if hasattr(backend, '_name'):
            backend._name = f"fake_{name}_v2"
        
        return backend
    
    def _initialize_simulator_backend(self):
        """Initialize a pure simulator backend with AerSimulator."""
        # Create AerSimulator instance
        self.backend = AerSimulator()
        self.backend_type = 'simulator'
        
        # Store backend characteristics for reference
        self.backend_info = {
            'name': 'aer_simulator',
            'num_qubits': 50,  # Arbitrary large number for simulator
            'basis_gates': ['id', 'rz', 'sx', 'x', 'cx', 'reset'],
            'coupling_map': [],  # Fully connected
            'simulator': True,
            'local': True,
            'backend_version': getattr(self.backend, 'backend_version', 'unknown')
        }
        
        # No service for simulator
        self._ibm_service = None
    
    def _get_coupling_map(self):
        """Extract coupling map from backend in a format consistent across versions."""
        try:
            # First try target-based approach (BackendV2)
            if hasattr(self.backend, 'target') and self.backend.target is not None:
                if hasattr(self.backend.target, 'coupling_map') and self.backend.target.coupling_map is not None:
                    cm = self.backend.target.coupling_map
                    if hasattr(cm, 'get_edges'):
                        return cm.get_edges()
                    elif hasattr(cm, 'get_connected_pairs'):
                        return cm.get_connected_pairs()
                    elif isinstance(cm, list):
                        return cm
                
                # Try to extract physical qubits from target
                if hasattr(self.backend.target, 'physical_qubits'):
                    physical_qubits = self.backend.target.physical_qubits
                    # Create a linear coupling map if we have physical qubits
                    if physical_qubits and len(physical_qubits) > 1:
                        return [[i, i+1] for i in range(len(physical_qubits) - 1)]
            
            # Try direct coupling_map attribute
            if hasattr(self.backend, 'coupling_map') and self.backend.coupling_map is not None:
                cm = self.backend.coupling_map
                if hasattr(cm, 'get_edges'):
                    return cm.get_edges()
                elif hasattr(cm, 'get_connected_pairs'):
                    return cm.get_connected_pairs()
                elif isinstance(cm, list):
                    return cm
                
            # Try configuration route (BackendV1 compatibility)
            if hasattr(self.backend, 'configuration'):
                config = self.backend.configuration()
                if hasattr(config, 'coupling_map') and config.coupling_map is not None:
                    return config.coupling_map
            
            # Fallback to creating a linear coupling map
            if hasattr(self.backend, 'num_qubits') and self.backend.num_qubits > 1:
                return [[i, i+1] for i in range(self.backend.num_qubits - 1)]
            
            # Last resort: return empty list with warning
            logger.warning("Could not determine coupling map, using empty map")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting coupling map: {str(e)}")
            # Fallback to linear coupling map
            if hasattr(self.backend, 'num_qubits') and self.backend.num_qubits > 1:
                return [[i, i+1] for i in range(self.backend.num_qubits - 1)]
            return []
    
    def register_reference_circuit(self, circuit: QuantumCircuit, label: Optional[str] = None) -> str:
        """
        Register a reference circuit and compute its ideal statevector.
        
        This stores a reference quantum state that can be used for error mitigation
        via pattern matching in the QSV frequency domain.
        
        For large circuits (>10 qubits), this method will automatically identify
        important subsystems and register a reduced reference state to improve
        memory efficiency.
        
        Args:
            circuit: QuantumCircuit to register
            label: Optional label (generated if None)
            
        Returns:
            Label of registered circuit
        """
        # Generate label if not provided
        if label is None:
            label = f"circuit_{len(self.reference_states) + 1}"
        
        # Create a clean copy of the circuit without measurements
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)
        
        # Handle large circuits differently
        num_qubits = circuit_copy.num_qubits
        max_full_statevector_qubits = self.config.get("max_reference_qubits", 10)
        
        if num_qubits > max_full_statevector_qubits:
            logger.info(f"Circuit has {num_qubits} qubits which exceeds threshold of {max_full_statevector_qubits}. Using selective registration.")
            return self._register_reduced_reference_circuit(circuit_copy, label, max_full_statevector_qubits)
        
        # Normal processing for smaller circuits
        try:
            # Try using Statevector class directly (preferred method)
            sv = Statevector.from_instruction(circuit_copy)
            statevector = sv.data
            logger.info(f"Successfully extracted statevector using Statevector class for '{label}'")
        except Exception as e1:
            logger.warning(f"Statevector extraction failed: {str(e1)}, falling back to simulator approach")
            
            # Use simulator approach as fallback
            simulator = AerSimulator(method='statevector')
            transpiled_circuit = transpile(circuit_copy, simulator)
            
            try:
                result = simulator.run(transpiled_circuit).result()
                statevector = result.get_statevector()
                logger.info(f"Successfully extracted statevector using simulator for '{label}'")
            except Exception as e2:
                error_msg = f"Failed to extract statevector for circuit {label}: {str(e2)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Store the reference statevector
        self.reference_states[label] = statevector
        
        # Register with QuantumEye
        self.quantum_eye.register_reference_state(label, statevector)
        
        return label
        
    def _register_reduced_reference_circuit(self, circuit: QuantumCircuit, label: str, max_qubits: int) -> str:
        """
        Register a reduced version of a large reference circuit by focusing on the most important qubits.
        
        Args:
            circuit: Original quantum circuit (without measurements)
            label: Label for the reference state
            max_qubits: Maximum number of qubits to include in the reduced circuit
            
        Returns:
            Label of registered reference circuit
        """
        # Identify important qubits in the circuit
        important_qubits = self._identify_important_qubits(circuit, max_qubits)
        
        # Create modified label to indicate this is a reduced reference
        reduced_label = f"{label}_reduced"
        
        try:
            # Create reduced statevector from important qubits
            reduced_statevector = self._get_reduced_statevector(circuit, important_qubits)
            logger.info(f"Generated reduced statevector for '{reduced_label}' using {len(important_qubits)} qubits")
            
            # Store the reduced reference statevector
            self.reference_states[reduced_label] = reduced_statevector
            
            # Register with QuantumEye
            self.quantum_eye.register_reference_state(reduced_label, reduced_statevector)
            
            return reduced_label
            
        except Exception as e:
            error_msg = f"Failed to create reduced reference state for {label}: {str(e)}"
            logger.error(error_msg)
            
            # Fall back to using a sparse approximation if possible
            try:
                logger.info("Attempting sparse approximation of statevector")
                sparse_sv = self._get_sparse_statevector_approximation(circuit)
                if sparse_sv is not None:
                    sparse_label = f"{label}_sparse"
                    self.reference_states[sparse_label] = sparse_sv
                    self.quantum_eye.register_reference_state(sparse_label, sparse_sv)
                    return sparse_label
            except Exception as e2:
                logger.error(f"Sparse approximation failed: {str(e2)}")
            
            # If all else fails, create a simple state for the important qubits
            logger.warning("Creating simple reference state for important qubits")
            simple_circuit = QuantumCircuit(len(important_qubits))
            simple_label = f"{label}_simple"
            
            # For GHZ-like states, create a simple GHZ state on important qubits
            if self._is_likely_ghz(circuit):
                simple_circuit.h(0)
                for i in range(1, len(important_qubits)):
                    simple_circuit.cx(0, i)
            
            # Get statevector for simple circuit
            sv = Statevector.from_instruction(simple_circuit)
            simple_sv = sv.data
            
            # Store simplified reference
            self.reference_states[simple_label] = simple_sv
            self.quantum_eye.register_reference_state(simple_label, simple_sv)
            
            return simple_label
            
    def _identify_important_qubits(self, circuit: QuantumCircuit, max_qubits: int) -> List[int]:
        """
        Identify the most important qubits in a circuit for reduced state representation.
        
        This method analyzes gate types and connectivity to find qubits that contribute
        most to the state's structure (e.g., qubits involved in entanglement).
        
        Args:
            circuit: Quantum circuit to analyze
            max_qubits: Maximum number of qubits to select
            
        Returns:
            List of qubit indices identified as most important
        """
        num_qubits = circuit.num_qubits
        
        # If we can include all qubits, do so
        if num_qubits <= max_qubits:
            return list(range(num_qubits))
            
        # Count operations per qubit
        qubit_importance = {i: 0 for i in range(num_qubits)}
        
        # Count connectivity (how many other qubits each qubit connects to)
        connectivity = {i: set() for i in range(num_qubits)}
        
        # Analyze circuit for qubit importance
        for instruction, qargs, _ in circuit.data:
            # Get instruction name
            inst_name = instruction.name
            
            # Get qubit indices - Fix: Use circuit.qubits.index() instead of qarg.index
            qubit_indices = [circuit.qubits.index(qarg) for qarg in qargs]
            
            # Weight multi-qubit gates higher (they create entanglement)
            if len(qubit_indices) > 1:
                weight = 3.0  # Higher weight for entangling gates
                
                # Special handling for specific gates
                if inst_name in ['cx', 'cz', 'cp']:
                    weight = 5.0  # Control gates are important
                elif inst_name in ['rzz', 'rxx', 'ryy']:
                    weight = 8.0  # Often critical for phase transitions
                elif inst_name in ['swap']:
                    weight = 4.0
                
                # Update connectivity
                for i, q1 in enumerate(qubit_indices):
                    for q2 in qubit_indices[i+1:]:
                        connectivity[q1].add(q2)
                        connectivity[q2].add(q1)
            else:
                weight = 1.0
                
                # Special handling for specific gates
                if inst_name in ['h', 's', 'sdg']:
                    weight = 2.0  # Phase-related gates
                elif inst_name in ['t', 'tdg']:
                    weight = 1.5
                    
            # Update importance scores
            for idx in qubit_indices:
                qubit_importance[idx] += weight
        
        # Add connectivity score
        for q, connected in connectivity.items():
            qubit_importance[q] += len(connected) * 2.0
            
        # Sort qubits by importance score
        sorted_qubits = sorted(qubit_importance.keys(), 
                              key=lambda q: qubit_importance[q], 
                              reverse=True)
        
        # Take the top max_qubits
        important_qubits = sorted_qubits[:max_qubits]
        
        # Ensure we include some qubits from the beginning, middle, and end for better coverage
        if num_qubits > 3 * max_qubits:
            if 0 not in important_qubits and len(important_qubits) > 0:
                important_qubits[-1] = 0  # Replace least important with first qubit
                
            middle_idx = num_qubits // 2
            if middle_idx not in important_qubits and len(important_qubits) > 1:
                important_qubits[-2] = middle_idx  # Replace second least important with middle qubit
                
            last_idx = num_qubits - 1
            if last_idx not in important_qubits and len(important_qubits) > 2:
                important_qubits[-3] = last_idx  # Replace third least important with last qubit
        
        # Sort the indices to maintain order
        important_qubits.sort()
        
        logger.info(f"Selected {len(important_qubits)} important qubits: {important_qubits}")
        return important_qubits
        
    def _get_reduced_statevector(self, circuit: QuantumCircuit, qubits: List[int]) -> np.ndarray:
        """
        Extract a reduced statevector for the specified subset of qubits.
        
        Args:
            circuit: Original quantum circuit
            qubits: List of qubit indices to include
            
        Returns:
            Reduced statevector focusing on selected qubits
        """
        if len(qubits) == 0:
            raise ValueError("Cannot extract reduced statevector with empty qubit list")
            
        # Create a new circuit with just the selected qubits
        reduced_circuit = QuantumCircuit(len(qubits))
        
        # Create mapping from original qubits to reduced circuit qubits
        qubit_mapping = {orig: idx for idx, orig in enumerate(qubits)}
        
        # Copy over operations that only involve selected qubits
        for instruction, qargs, cargs in circuit.data:
            # Get qubit indices - Fix: Use circuit.qubits.index() instead of qarg.index
            qubit_indices = [circuit.qubits.index(qarg) for qarg in qargs]
            
            # Only include operations where all qubits are in our selected list
            if all(idx in qubits for idx in qubit_indices):
                # Map original qubit indices to new indices
                new_qargs = [reduced_circuit.qubits[qubit_mapping[idx]] for idx in qubit_indices]
                
                # Add instruction to reduced circuit
                reduced_circuit.append(instruction, new_qargs, [])
        
        # Get statevector of reduced circuit
        try:
            # Try using Statevector class directly
            sv = Statevector.from_instruction(reduced_circuit)
            return sv.data
        except Exception as e:
            logger.warning(f"Error computing reduced statevector: {str(e)}")
            
            # Fallback to simulator approach
            simulator = AerSimulator(method='statevector')
            transpiled_circuit = transpile(reduced_circuit, simulator)
            result = simulator.run(transpiled_circuit).result()
            return result.get_statevector()
            
    def _get_sparse_statevector_approximation(self, circuit: QuantumCircuit, threshold: float = 1e-6) -> np.ndarray:
        """
        Get a sparse approximation of the statevector by setting small amplitudes to zero.
        
        Args:
            circuit: Quantum circuit
            threshold: Amplitude threshold below which values are set to zero
            
        Returns:
            Sparse approximation of the statevector
        """
        try:
            # Try to get statevector using sampler with many shots
            sampler = AerSamplerV2(run_options={"shots": 20000})
            job = sampler.run([circuit])
            result = job.result()
            pub_result = result[0];
            
            # Extract counts
            if hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()
            else:
                first_reg = next(iter(pub_result.data))
                counts = pub_result.data[first_reg].get_counts()
                
            # Convert counts to sparse statevector
            dim = 2**circuit.num_qubits
            sparse_sv = np.zeros(dim, dtype=complex)
            total_shots = sum(counts.values())
            
            # Only include states with significant probability
            for bitstring, count in counts.items():
                # Convert bitstring to index (reversing bit order for endianness)
                idx = int(bitstring, 2)
                
                # Set amplitude (square root of probability, with random phase)
                prob = count / total_shots
                if prob > threshold:
                    amplitude = np.sqrt(prob)
                    # Add random phase
                    angle = np.random.uniform(0, 2*np.pi)
                    sparse_sv[idx] = amplitude * np.exp(1j * angle)
            
            # Normalize
            norm = np.linalg.norm(sparse_sv)
            if norm > 0:
                sparse_sv = sparse_sv / norm
                
            return sparse_sv
            
        except Exception as e:
            logger.error(f"Failed to create sparse statevector: {str(e)}")
            return None
            
    def _is_likely_ghz(self, circuit: QuantumCircuit) -> bool:
        """
        Check if a circuit is likely to be a GHZ state preparation.
        
        Args:
            circuit: Circuit to analyze
            
        Returns:
            Boolean indicating if the circuit likely prepares a GHZ state
        """
        # Look for pattern: H on first qubit followed by CNOTs to other qubits
        has_initial_h = False
        cnot_count = 0
        
        for instruction, qargs, _ in circuit.data:
            # Fix: Use circuit.qubits.index() instead of qarg.index
            if instruction.name == 'h' and circuit.qubits.index(qargs[0]) == 0:
                has_initial_h = True
                
            if instruction.name == 'cx' and circuit.qubits.index(qargs[0]) == 0:
                cnot_count += 1
                
        # If we have initial H and CNOTs from qubit 0 to others, likely a GHZ
        return has_initial_h and cnot_count > 0
    
    def _is_likely_zero_state(self, circuit):
        """
        Check if circuit is likely to produce |00...0⟩ state
        Simple heuristic: no gates applied = zero state
        """
        # Count non-measurement operations
        gate_count = 0
        for instruction in circuit.data:
            if instruction.operation.name not in ['measure', 'barrier']:
                gate_count += 1
        
        # If very few gates, likely zero state
        return gate_count <= 1
    
    def _prepare_zero_state_circuit(self, circuit, noise_level):
        """Prepare a modified circuit for zero state with small rotation"""
        modified_circuit = circuit.copy()
        # Add small rotation to ensure noise visibility
        if noise_level > 0:
            modified_circuit.ry(0.01, 0)  # Small rotation on first qubit
        return modified_circuit
    
    def execute_circuit(
        self, 
        circuit: QuantumCircuit, 
        shots: Optional[int] = None,
        mitigation_enabled: bool = True,
        reference_label: Optional[str] = None,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        use_error_map: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a circuit on the configured backend (real, fake, or simulator).
        
        Args:
            circuit: QuantumCircuit to execute
            shots: Number of shots for the execution
            mitigation_enabled: Whether to apply Quantum Eye mitigation
            reference_label: Optional label of reference state to use
            noise_type: Type of noise model to use (for simulators)
            noise_level: Noise level parameter (for simulators)
            use_error_map: Whether to use backend error map for mitigation
            
        Returns:
            Dictionary with execution results
        """
        # Get configuration values with defaults
        shots = shots or self.config.get("default_shots", 1024)
        noise_type = noise_type or self.config.get("noise_type", "depolarizing")
        noise_level = noise_level if noise_level is not None else self.config.get("noise_level", 0.1)
        
        # Ensure circuit has measurements
        measuring_circuit = circuit.copy()
        if not measuring_circuit.num_clbits:
            measuring_circuit.measure_all()
        
        # Execute based on backend type
        if self.backend_type == 'real':
            return self._execute_on_real_backend(
                measuring_circuit, shots, mitigation_enabled, reference_label, use_error_map)
        else:
            return self._execute_on_simulator(
                measuring_circuit, shots, mitigation_enabled, 
                reference_label, noise_type, noise_level, use_error_map)
    
    def _execute_on_real_backend(
        self, 
        circuit: QuantumCircuit,
        shots: int,
        mitigation_enabled: bool,
        reference_label: Optional[str],
        use_error_map: bool
    ) -> Dict[str, Any]:
        """
        Execute circuit on real IBM Quantum hardware using SamplerV2.
        
        This method is specifically for measurement-based execution (counts),
        NOT for expectation value calculations (use _estimator_on_real_backend for that).
        
        Args:
            circuit: QuantumCircuit to execute (must have measurements)
            shots: Number of shots
            mitigation_enabled: Whether to apply Quantum Eye mitigation
            reference_label: Optional reference state label
            use_error_map: Whether to use backend error map for mitigation
            
        Returns:
            Dictionary with execution results including counts
        """
        if not HAS_IBM_RUNTIME or not hasattr(self, '_ibm_service'):
            raise RuntimeError("IBM Quantum Runtime not available")
        
        try:
            # Ensure circuit has measurements
            if not circuit.num_clbits:
                logger.warning("Circuit has no measurements, adding measure_all()")
                circuit = circuit.copy()
                circuit.measure_all()
            
            # Transpile circuit for the backend
            transpiled_circuit = self._transpile_for_backend(circuit, self.backend)
            
            # Use Batch for execution
            with Batch(backend=self.backend) as batch:
                # Configure sampler options - FIXED: Use default_shots
                sampler_options = {
                    "default_shots": shots,
                    "twirling": {"enable_measure": True}
                }
                
                # Create SamplerV2 instance
                sampler = SamplerV2(mode=batch, options=sampler_options)
                
                # Submit job with PUB format (list of circuits)
                job = sampler.run([transpiled_circuit])
                
                # Wait for job completion
                self._wait_for_job(job)
                
                # Get result
                result = job.result()
                pub_result = result[0]  # First (and only) PubResult
                
                # Extract counts from the result
                if hasattr(pub_result.data, 'meas'):
                    # Standard measurement register
                    counts = pub_result.data.meas.get_counts()
                elif hasattr(pub_result.data, 'c'):
                    # Classical register named 'c'
                    counts = pub_result.data.c.get_counts()
                elif len(pub_result.data) > 0:
                    # Get the first available register
                    first_reg = next(iter(pub_result.data))
                    counts = pub_result.data[first_reg].get_counts()
                else:
                    # Handle empty result
                    logger.error("No measurement data found in result")
                    return self._handle_empty_result(circuit, shots)
                
                logger.info(f"Job {job.job_id()} completed successfully with {len(counts)} unique outcomes")
                
                # Apply Quantum Eye mitigation if enabled
                if mitigation_enabled:
                    logger.info("Applying Quantum Eye mitigation to counts")
                    mitigation_result = self._apply_mitigation_to_counts(
                        counts, circuit, reference_label)
                    
                    return {
                        "counts": counts,
                        "mitigation_result": mitigation_result,
                        "shots": shots,
                        "circuit": circuit,
                        "transpiled_circuit": transpiled_circuit,
                        "backend": self.backend.name,
                        "backend_type": self.backend_type,
                        "job_id": job.job_id(),
                        "mitigation_enabled": True
                    }
                else:
                    return {
                        "counts": counts,
                        "shots": shots,
                        "circuit": circuit,
                        "transpiled_circuit": transpiled_circuit,
                        "backend": self.backend.name,
                        "backend_type": self.backend_type,
                        "job_id": job.job_id(),
                        "mitigation_enabled": False
                    }
                    
        except Exception as e:
            error_msg = f"Real hardware execution failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _handle_empty_result(self, circuit: QuantumCircuit, shots: int) -> Dict[str, Any]:
        """
        Handle cases where no measurement data is returned.
        
        Args:
            circuit: Original circuit
            shots: Number of shots
            
        Returns:
            Default result dictionary
        """
        logger.warning("No measurement data found, creating default result")
        
        # Create a default result with all-zeros state
        default_counts = {'0' * circuit.num_qubits: shots}
        
        return {
            "counts": default_counts,
            "shots": shots,
            "circuit": circuit,
            "backend": self.backend.name,
            "backend_type": self.backend_type,
            "error": "No measurement data returned",
            "mitigation_enabled": False
        }
    
    def _execute_on_simulator(
        self,
        circuit: QuantumCircuit,
        shots: int,
        mitigation_enabled: bool,
        reference_label: Optional[str],
        noise_type: str,
        noise_level: float,
        use_error_map: bool
    ) -> Dict[str, Any]:
        """
        Execute circuit on simulator (either fake backend or pure simulator).
        
        Args:
            circuit: QuantumCircuit to execute
            shots: Number of shots
            mitigation_enabled: Whether to apply Quantum Eye mitigation
            reference_label: Optional reference state label
            noise_type: Type of noise model to use
            noise_level: Noise level parameter
            use_error_map: Whether to use backend error map for mitigation
            
        Returns:
            Dictionary with execution results
        """
        # Special handling for |0⟩ state circuits
        is_zero_state = self._is_likely_zero_state(circuit)
        if is_zero_state and noise_level > 0:
            # Create modified circuit with explicit reset and small rotation
            modified_circuit = self._prepare_zero_state_circuit(circuit, noise_level)
            # Add measurements to all qubits
            modified_circuit.measure_all()
            circuit = modified_circuit
            
        # Get or create noise model if noise is enabled
        noise_model = None
        if noise_level > 0:
            noise_model = self._get_noise_model(noise_type, noise_level)
        
        # Create options for AerSamplerV2 - Remove seed_simulator entirely
        options = {}
        if noise_model:
            options = {
                "backend_options": {
                    "noise_model": noise_model
                },
                "run_options": {
                    "shots": shots
                }
            }
        else:
            options = {
                "run_options": {
                    "shots": shots
                }
            }
        
        # Create the appropriate sampler based on backend type
        sampler = AerSamplerV2(options=options)
        
        try:
            # Run the circuit - Don't include seed_simulator parameter
            job = sampler.run([circuit])
            result = job.result()
            pub_result = result[0]  # Access first PubResult
            
            # Try to access the default measurement register
            if hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()
            # Try to access any available register
            elif len(pub_result.data) > 0:
                # Get the first available register name
                first_reg = next(iter(pub_result.data))
                counts = pub_result.data[first_reg].get_counts()
            else:
                # Handle cases where expected registers aren't found
                return self._handle_empty_result(circuit, shots)
            
            logger.info(f"Circuit executed successfully on simulator with {shots} shots")
            
            # Calculate ideal probabilities
            ideal_probs = self._calculate_ideal_probabilities(circuit)
            
            # Apply Quantum Eye mitigation if enabled
            if mitigation_enabled:
                mitigation_result = self._apply_mitigation_to_counts(
                    counts, circuit, reference_label)
                
                return {
                    "counts": counts,
                    "ideal_probs": ideal_probs,
                    "mitigation_result": mitigation_result,
                    "noise_model": noise_model,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "shots": shots,
                    "circuit": circuit,
                    "backend": self.backend.name,
                    "backend_type": self.backend_type
                }
            else:
                return {
                    "counts": counts,
                    "ideal_probs": ideal_probs,
                    "noise_model": noise_model,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "shots": shots,
                    "circuit": circuit,
                    "backend": self.backend.name,
                    "backend_type": self.backend_type
                }
                
        except Exception as e:
            error_msg = f"Simulator execution failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _transpile_for_backend(self, circuit: QuantumCircuit, backend: Any) -> QuantumCircuit:
        """
        Transpile a circuit for the target backend.
        
        Note: When transpiling for real backends, Qiskit allocates all physical qubits
        (e.g., 127 for IBM Brisbane). This is expected behavior, not a bug.
        
        Args:
            circuit: Circuit to transpile
            backend: Target backend
            
        Returns:
            Transpiled circuit
        """
        try:
            optimization_level = self.config.get('optimization_level', 1)
            
            logger.info(f"Transpiling {circuit.num_qubits}-qubit circuit for {backend.name}")
            
            # Just use standard transpilation - it works correctly
            transpiled_circuit = transpile(
                circuit, 
                backend=backend, 
                optimization_level=optimization_level
            )
            
            logger.info(f"Transpiled to {transpiled_circuit.num_qubits} qubits, depth {transpiled_circuit.depth()}")
            
            # Note: For real backends, the circuit will use all physical qubits.
            # This is normal and doesn't affect execution time or queue position.
            
            return transpiled_circuit
            
        except Exception as e:
            logger.error(f"Transpilation failed: {str(e)}")
            logger.info("Attempting fallback with optimization_level=0")
            
            try:
                # Try with minimal optimization
                return transpile(circuit, backend=backend, optimization_level=0)
            except Exception as e2:
                logger.error(f"Fallback transpilation also failed: {str(e2)}")
                raise
    
    def _wait_for_job(self, job, timeout=None):
        """
        Wait for job completion with status updates.
        
        Args:
            job: RuntimeJobV2 instance
            timeout: Maximum wait time in seconds (None for no timeout)
        """
        start_time = time.time()
        status_check_interval = 5  # seconds
        
        logger.info(f"Waiting for job {job.job_id()} to complete...")
        
        try:
            prev_status = None
            
            while True:
                # Check current status
                current_status = job.status()
                
                # Only log status changes
                if current_status != prev_status:
                    logger.info(f"Job status: {current_status}")
                    prev_status = current_status
                
                # Check if job is done or failed
                if current_status == 'DONE':
                    logger.info(f"Job {job.job_id()} completed successfully after {time.time() - start_time:.1f} seconds")
                    return
                elif current_status == 'ERROR':
                    error_msg = job.error_message()
                    raise RuntimeError(f"Job failed: {error_msg}")
                elif current_status == 'CANCELLED':
                    raise RuntimeError("Job was cancelled")
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Timeout after waiting {timeout} seconds for job completion")
                
                # Wait before checking again
                time.sleep(status_check_interval)
                
        except Exception as e:
            logger.error(f"Error monitoring job: {str(e)}")
            raise
    
    def _apply_mitigation_to_counts(
        self, 
        counts: Dict[str, int], 
        circuit: QuantumCircuit,
        reference_label: Optional[str],
        component_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Apply Quantum Eye mitigation to measurement counts.
        
        Args:
            counts: Measurement counts dictionary
            circuit: Original circuit
            reference_label: Optional reference state label
            component_weights: Optional weights for QSV components
            
        Returns:
            Dictionary with mitigation results
        """
        # Enforce memory constraints based on circuit size
        num_qubits = circuit.num_qubits
        max_full_transform_qubits = self.config.get("max_transform_qubits", 6)
        
        if num_qubits > max_full_transform_qubits:
            logger.warning(f"Circuit has {num_qubits} qubits which exceeds mitigation threshold of {max_full_transform_qubits}.")
            logger.info("Skipping mitigation for large system - returning counts directly")
            
            # For large systems, skip mitigation entirely and return counts directly
            return {
                "mitigated": False,
                "mitigated_counts": counts,
                "original_counts": counts,
                "reference_label": reference_label,
                "mitigation_method": "none_large_system",
                "reason": f"System too large ({num_qubits} > {max_full_transform_qubits} qubits)",
                "improvement": 0.0
            }
        
        # Standard mitigation for smaller circuits
        # Convert counts to approximate statevector
        noisy_statevector = self._statevector_from_counts(counts, num_qubits)
        
        # Check if statevector reconstruction failed (e.g., system too large)
        if noisy_statevector is None:
            logger.warning(f"Cannot reconstruct statevector for {num_qubits} qubits - system too large")
            return {
                "mitigated": False,
                "mitigated_counts": counts,
                "original_counts": counts,
                "reference_label": reference_label,
                "mitigation_method": "none_statevector_too_large",
                "reason": f"Cannot reconstruct statevector for {num_qubits} qubits",
                "improvement": 0.0
            }
        
        # Find best reference if not specified
        if reference_label is None:
            reference_label = self._find_best_reference_match(noisy_statevector)
            logger.info(f"Auto-selected reference: {reference_label}")
            
        # Apply mitigation with or without reference
        if reference_label and reference_label in self.reference_states:
            logger.info("Applying mitigation with reference state")
            mitigation_result = self.quantum_eye.mitigate(
                noisy_statevector, 
                reference_state=self.reference_states[reference_label],
                component_weights=component_weights
            )
        else:
            logger.info("Applying mitigation without reference state")
            mitigation_result = self.quantum_eye.mitigate(
                noisy_statevector,
                component_weights=component_weights
            )
        
        # Calculate improvement metrics
        if reference_label and reference_label in self.reference_states:
            ideal_statevector = self.reference_states[reference_label]
            mitigated_state = mitigation_result["mitigated_state"]
            
            # Calculate fidelities
            original_fidelity = abs(np.vdot(noisy_statevector, ideal_statevector))**2
            mitigated_fidelity = abs(np.vdot(mitigated_state, ideal_statevector))**2
            
            # Calculate improvement (handling special cases)
            if original_fidelity >= 0.999:
                # Already perfect, no improvement possible
                improvement = 0.0
            else:
                # Calculate relative improvement
                improvement = (mitigated_fidelity - original_fidelity) / (1.0 - original_fidelity)
                # Ensure valid range
                improvement = min(1.0, max(0.0, improvement))
                
            mitigation_result["original_fidelity"] = float(original_fidelity)
            mitigation_result["mitigated_fidelity"] = float(mitigated_fidelity)
            mitigation_result["improvement"] = float(improvement)
            
            logger.info(f"Mitigation complete - Original fidelity: {original_fidelity:.4f}, " 
                       f"Mitigated fidelity: {mitigated_fidelity:.4f}, "
                       f"Improvement: {improvement:.4f}")
        
        # Calculate mitigated counts for easier comparison
        mitigated_counts = self._counts_from_statevector(
            mitigation_result["mitigated_state"],
            circuit.num_qubits,
            sum(counts.values())
        )
        mitigation_result["mitigated_counts"] = mitigated_counts
        mitigation_result["reference_label"] = reference_label
        mitigation_result["mitigated"] = True
        
        return mitigation_result
    
    def _apply_chunked_mitigation(
        self,
        counts: Dict[str, int],
        circuit: QuantumCircuit,
        reference_label: Optional[str],
        component_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Apply mitigation to large circuits by processing in manageable chunks.
        
        For large circuits, we identify important qubits and focus mitigation there
        to reduce memory requirements.
        
        Args:
            counts: Measurement counts dictionary
            circuit: Original circuit
            reference_label: Optional reference state label
            component_weights: Optional weights for QSV components
            
        Returns:
            Dictionary with mitigation results
        """
        num_qubits = circuit.num_qubits
        logger.info(f"Using chunked mitigation approach for {num_qubits}-qubit circuit")
        
        # Get config settings
        max_chunk_size = self.config.get("max_chunk_qubits", 12)
        
        # Identify important qubits in the circuit
        important_qubits = self._identify_important_qubits(circuit, max_chunk_size)
        
        # Create a reduced reference state if one doesn't exist
        reduced_ref_label = None
        if reference_label:
            reduced_ref_label = f"{reference_label}_reduced"
            if reduced_ref_label not in self.reference_states and reference_label in self.reference_states:
                # Create reduced reference state
                full_ref_state = self.reference_states[reference_label]
                reduced_ref_state = self._reduce_statevector_to_qubits(full_ref_state, num_qubits, important_qubits)
                self.reference_states[reduced_ref_label] = reduced_ref_state
                self.quantum_eye.register_reference_state(reduced_ref_label, reduced_ref_state)
        
        # Extract only the important qubits from the counts
        reduced_counts = self._reduce_counts_to_qubits(counts, important_qubits)
        
        # Convert reduced counts to statevector
        reduced_statevector = self._statevector_from_counts(reduced_counts, len(important_qubits))
        
        # Apply mitigation to reduced statevector
        if reduced_ref_label and reduced_ref_label in self.reference_states:
            logger.info(f"Applying mitigation with reduced reference state {reduced_ref_label}")
            reduced_result = self.quantum_eye.mitigate(
                reduced_statevector,
                reference_state=self.reference_states[reduced_ref_label],
                component_weights=component_weights
            )
        else:
            logger.info("Applying mitigation without reference state")
            reduced_result = self.quantum_eye.mitigate(
                reduced_statevector,
                component_weights=component_weights
            )
        
        # Calculate metrics and prepare result
        mitigated_result = {
            "mitigated": True,
            "reference_label": reduced_ref_label or reference_label,
            "important_qubits": important_qubits,
            "mitigation_method": "chunked_quantum_eye",
            "original_counts": counts
        }
        
        # Convert mitigated reduced statevector back to counts
        mitigated_reduced_counts = self._counts_from_statevector(
            reduced_result["mitigated_state"],
            len(important_qubits),
            sum(reduced_counts.values())
        )
        mitigated_result["mitigated_reduced_counts"] = mitigated_reduced_counts
        
        # Expand reduced counts to full counts
        mitigated_full_counts = self._expand_counts_from_qubits(
            mitigated_reduced_counts, counts, important_qubits, num_qubits
        )
        mitigated_result["mitigated_counts"] = mitigated_full_counts
        
        # Calculate fidelity if reference is available
        if reduced_ref_label and reduced_ref_label in self.reference_states:
            ideal_reduced = self.reference_states[reduced_ref_label]
            mitigated_reduced = reduced_result["mitigated_state"]
            
            original_fidelity = abs(np.vdot(reduced_statevector, ideal_reduced))**2
            mitigated_fidelity = abs(np.vdot(mitigated_reduced, ideal_reduced))**2
            
            # Calculate improvement
            if original_fidelity >= 0.999:
                improvement = 0.0
            else:
                improvement = (mitigated_fidelity - original_fidelity) / (1.0 - original_fidelity)
                improvement = min(1.0, max(0.0, improvement))
            
            mitigated_result["original_fidelity"] = float(original_fidelity)
            mitigated_result["mitigated_fidelity"] = float(mitigated_fidelity)
            mitigated_result["improvement"] = float(improvement)
            
            logger.info(f"Chunked mitigation complete - Original fidelity: {original_fidelity:.4f}, " 
                       f"Mitigated fidelity: {mitigated_fidelity:.4f}, "
                       f"Improvement: {improvement:.4f}")
        else:
            logger.info("Chunked mitigation complete - No reference available for fidelity calculation")
        
        return mitigated_result
    
    def _reduce_statevector_to_qubits(self, 
                                    statevector: np.ndarray, 
                                    num_qubits: int, 
                                    important_qubits: List[int]) -> np.ndarray:
        """
        Reduce a statevector to focus on specific important qubits.
        
        Args:
            statevector: Full statevector
            num_qubits: Total number of qubits
            important_qubits: List of important qubit indices to keep
            
        Returns:
            Reduced statevector for important qubits
        """
        # Check if already appropriately sized
        if len(statevector) == 2**len(important_qubits):
            return statevector
        
        # Create mapping for bits we want to keep
        keep_mapping = {q: i for i, q in enumerate(important_qubits)}
        
        # Create reduced statevector
        reduced_dim = 2**len(important_qubits)
        reduced_sv = np.zeros(reduced_dim, dtype=complex)
        
        # Process each amplitude in original statevector
        for idx, amplitude in enumerate(statevector):
            if abs(amplitude) < 1e-10:
                continue
                
            # Convert index to bit string
            bitstring = format(idx, f'0{num_qubits}b')
            
            # Extract bits for important qubits
            reduced_bits = ''.join(bitstring[num_qubits - 1 - q] for q in important_qubits)
            
            # Convert back to index
            reduced_idx = int(reduced_bits, 2)
            
            # Accumulate probability (not amplitude)
            reduced_sv[reduced_idx] += abs(amplitude)**2
        
        # Take square root to get amplitudes, using uniform phases
        for i in range(reduced_dim):
            if reduced_sv[i] > 0:
                reduced_sv[i] = np.sqrt(reduced_sv[i])
        
        # Normalize
        norm = np.linalg.norm(reduced_sv)
        if norm > 0:
            reduced_sv = reduced_sv / norm
        
        return reduced_sv
    
    def _reduce_counts_to_qubits(self, counts: Dict[str, int], important_qubits: List[int]) -> Dict[str, int]:
        """
        Reduce measurement counts to focus only on important qubits.
        
        Args:
            counts: Original counts dictionary
            important_qubits: List of important qubit indices to keep
            
        Returns:
            Reduced counts dictionary focusing only on important qubits
        """
        reduced_counts = {}
        
        # Get total number of qubits from first key
        for bitstring in counts.keys():
            num_qubits = len(bitstring)
            break
        
        # Process each measurement outcome
        for bitstring, count in counts.items():
            # Make sure bitstring has the right length
            bitstring = bitstring.zfill(num_qubits)
            
            # Extract bits for important qubits
            # Note: Bitstrings are in reverse order compared to qubit indices
            reduced_bits = ''.join(bitstring[num_qubits - 1 - q] for q in important_qubits)
            
            # Add to reduced counts
            if reduced_bits in reduced_counts:
                reduced_counts[reduced_bits] += count
            else:
                reduced_counts[reduced_bits] = count
        
        return reduced_counts
    
    def _expand_counts_from_qubits(self, 
                                 reduced_counts: Dict[str, int],
                                 original_counts: Dict[str, int],
                                 important_qubits: List[int],
                                 num_qubits: int) -> Dict[str, int]:
        """
        Expand reduced counts back to full counts using original distribution pattern.
        
        Args:
            reduced_counts: Counts for important qubits only
            original_counts: Original full counts
            important_qubits: List of important qubit indices
            num_qubits: Total number of qubits
            
        Returns:
            Expanded counts dictionary
        """
        # Create pattern map from original counts
        pattern_map = {}
        total_counts = {}
        
        # First pass: create mapping from reduced to full patterns and count totals
        for bitstring, count in original_counts.items():
            # Make sure bitstring has the right length
            bitstring = bitstring.zfill(num_qubits)
            
            # Extract bits for important qubits
            reduced_bits = ''.join(bitstring[num_qubits - 1 - q] for q in important_qubits)
            
            # Add pattern to map
            if reduced_bits not in pattern_map:
                pattern_map[reduced_bits] = []
                total_counts[reduced_bits] = 0
            
            pattern_map[reduced_bits].append(bitstring)
            total_counts[reduced_bits] += count
        
        # Second pass: expand reduced counts back to full counts
        expanded_counts = {}
        
        for reduced_bits, count in reduced_counts.items():
            if reduced_bits in pattern_map:
                patterns = pattern_map[reduced_bits]
                original_total = total_counts[reduced_bits]
                
                # Distribute the count proportionally based on original pattern
                for pattern in patterns:
                    if original_total > 0:
                        original_fraction = original_counts[pattern.lstrip('0')] / original_total
                        expanded_counts[pattern.lstrip('0')] = int(round(count * original_fraction))
                    else:
                        # If no original counts, distribute evenly
                        expanded_counts[pattern.lstrip('0')] = int(round(count / len(patterns)))
            else:
                # No matching pattern in original counts
                # Create a reasonable default pattern
                default_pattern = ['0'] * num_qubits
                for i, q in enumerate(important_qubits):
                    default_pattern[num_qubits - 1 - q] = reduced_bits[i]
                expanded_counts[''.join(default_pattern).lstrip('0')] = count
        
        # Ensure we didn't lose any counts due to rounding
        total_expanded = sum(expanded_counts.values())
        total_reduced = sum(reduced_counts.values())
        
        if total_expanded != total_reduced and total_expanded > 0:
            # Find the highest count and adjust it
            max_pattern = max(expanded_counts, key=expanded_counts.get)
            expanded_counts[max_pattern] += (total_reduced - total_expanded)
        
        return expanded_counts
    
    def _find_best_reference_match(self, state: np.ndarray) -> Optional[str]:
        """
        Find the best matching reference state based on fidelity.
        
        Args:
            state: Quantum state to match
            
        Returns:
            Label of best matching reference state or None
        """
        # Check if we have reference states available
        if not self.reference_states:
            logger.warning("No reference states available for matching")
            return None
            
        best_match = None
        best_fidelity = 0.0
        # Get matching threshold from config
        threshold = self.config.get('reference_match_threshold', 0.6)
        
        # Compare the state against each reference
        for label, ref_state in self.reference_states.items():
            # Ensure state and ref_state have compatible dimensions
            if len(state) != len(ref_state):
                logger.warning(f"Dimension mismatch between state ({len(state)}) and reference {label} ({len(ref_state)})")
                continue
                
            # Calculate fidelity
            try:
                fidelity = abs(np.vdot(state, ref_state))**2
                
                # Update best match if needed
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_match = label
                    
                logger.debug(f"Reference '{label}' has fidelity {fidelity:.4f}")
            except Exception as e:
                logger.warning(f"Error calculating fidelity with reference '{label}': {str(e)}")
        
        # Only return a match if it exceeds the threshold
        if best_match and best_fidelity >= threshold:
            logger.info(f"Best reference match: {best_match} with fidelity {best_fidelity:.4f}")
            return best_match
        else:
            logger.info(f"No reference match found above threshold {threshold} (best: {best_fidelity:.4f})")
            return None
    
    def execute_with_estimator(
        self, 
        circuit: QuantumCircuit, 
        observable: Any, 
        parameter_values: Optional[Union[List[float], np.ndarray]] = None,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        mitigation_enabled: bool = True,
        shots: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute circuit with Estimator primitive and apply mitigation.
        
        This method works with both real hardware and simulators.
        
        Args:
            circuit: QuantumCircuit to execute
            observable: Observable to measure (SparsePauliOp)
            parameter_values: Optional values for circuit parameters
            noise_type: Type of noise model to use
            noise_level: Noise level parameter
            mitigation_enabled: Whether to apply Quantum Eye mitigation
            shots: Number of shots for simulation
            
        Returns:
            Dictionary with estimation results
        """
        # Get configuration values with defaults
        shots = shots or self.config.get("default_shots", 1024)
        noise_type = noise_type or self.config.get("noise_type", "depolarizing")
        noise_level = noise_level if noise_level is not None else self.config.get("noise_level", 0.1)
        
        # Create a copy without measurements (for Estimator)
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)
        
        # Execute based on backend type
        if self.backend_type == 'real':
            return self._estimator_on_real_backend(
                circuit_copy, observable, parameter_values, 
                shots, mitigation_enabled)
        else:
            return self._estimator_on_simulator(
                circuit_copy, observable, parameter_values, 
                noise_type, noise_level, shots, mitigation_enabled)
    
    def _estimator_on_real_backend(
        self,
        circuit: QuantumCircuit,
        observable: Any,
        parameter_values: Optional[Union[List[float], np.ndarray]],
        shots: int,
        mitigation_enabled: bool
    ) -> Dict[str, Any]:
        """
        Execute circuit with Estimator on real hardware.
        
        Args:
            circuit: QuantumCircuit to execute
            observable: Observable to measure
            parameter_values: Optional parameter values
            shots: Number of shots
            mitigation_enabled: Whether to apply Quantum Eye mitigation
            
        Returns:
            Dictionary with estimation results
        """
        # IBM Runtime path (original code)
        if not HAS_IBM_RUNTIME or not hasattr(self, '_ibm_service'):
            raise RuntimeError("IBM Quantum Runtime not available")
        
        try:
            # Transpile circuit to ISA
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            isa_circuit = pm.run(circuit)
            
            # CRITICAL: Apply layout to observable to match transpiled circuit
            isa_observable = observable.apply_layout(isa_circuit.layout)
            
            # Create options - FIXED: Use default_shots
            options = EstimatorOptions()
            options.default_shots = shots
            options.resilience_level = 1
            
            # Create a Batch object
            batch = Batch(backend=self.backend)
            
            # Create estimator with correct PUB format
            estimator = EstimatorV2(mode=batch, options=options)
            
            # Submit job - PUB format: list of tuples
            if parameter_values is not None:
                job = estimator.run([(isa_circuit, isa_observable, parameter_values)])
            else:
                job = estimator.run([(isa_circuit, isa_observable)])
            
            # Wait for job and get result
            result = job.result()
            expectation_value = float(result[0].data.evs)
            
            return {
                'expectation_value': expectation_value,
                'mitigated_expectation': expectation_value,
                'job_id': job.job_id(),
                'backend': self.backend.name,
                'shots': shots
            }
    
        except Exception as e:
            error_msg = f"Real hardware estimator execution failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def get_backend_info(self):
        """Get backend information dictionary."""
        return self.backend_info

    def _statevector_from_counts(self, counts, num_qubits):
        """
        Reconstruct statevector from measurement counts using maximum likelihood estimation.
        
        NOTE: This method reconstructs quantum state amplitudes purely from measurement 
        statistics without using prior knowledge of the ideal state. The reconstruction
        assumes real-valued amplitudes (no phase information) since measurement counts
        only provide probability information.
        
        Memory considerations:
        - Works well for small systems (≤6 qubits, 64 amplitudes)
        - For larger systems (>6 qubits), consider returning None and working 
        directly with counts to avoid memory issues
        - Alternative: Use compressed representations for sparse states
        
        Args:
            counts: Dictionary of measurement counts (e.g., {'00': 512, '01': 256, ...})
            num_qubits: Number of qubits in the system
            
        Returns:
            numpy.ndarray: Reconstructed statevector with real-valued amplitudes,
                        or None for large systems to avoid memory constraints
        """
        # Check against configured threshold (default 6, but can be increased)
        max_transform_qubits = self.config.get("max_transform_qubits", 6)
        if num_qubits > max_transform_qubits:  
            return None  # Return None, work directly with counts
        total_shots = sum(counts.values())
        num_states = 2 ** num_qubits
        
        # Initialize statevector with uniform amplitudes
        statevector = np.zeros(num_states, dtype=complex)
        
        # Convert counts to probabilities and take square root for amplitudes
        for state_str, count in counts.items():
            # Convert binary string to integer index
            state_index = int(state_str, 2)
            probability = count / total_shots
            # Use square root of probability as amplitude (real-valued approximation)
            statevector[state_index] = np.sqrt(probability)
        
        # Normalize the statevector
        norm = np.linalg.norm(statevector)
        if norm > 0:
            statevector = statevector / norm
        
        return statevector
    
    def _counts_from_statevector(self, statevector, num_qubits, shots):  
        """
        Convert statevector to measurement counts using probabilistic sampling.
        
        Args:
            statevector: Quantum state vector
            num_qubits: Number of qubits  
            shots: Number of measurement shots to simulate
            
        Returns:
            dict: Measurement counts dictionary
        """
        # Get probabilities from statevector
        probs = np.abs(statevector) ** 2
        # num_qubits = int(np.log2(len(statevector)))  # use parameter instead
        
        # Generate random samples based on probabilities
        samples = np.random.choice(len(probs), size=shots, p=probs)
        
        # Convert to binary strings and count
        counts = {}
        for sample in samples:
            # Convert integer to binary string
            binary_str = format(sample, f'0{num_qubits}b')
            counts[binary_str] = counts.get(binary_str, 0) + 1
        
        return counts

    def _get_noise_model(self, noise_type: str, noise_level: float):
        """
        Get or create a noise model for simulation.
        """
        if noise_level <= 0:
            return None
        
        # Check cache first
        cache_key = f"{noise_type}_{noise_level}"
        if hasattr(self, '_noise_model_cache') and cache_key in self._noise_model_cache:
            return self._noise_model_cache[cache_key]
        
        # Initialize cache if needed
        if not hasattr(self, '_noise_model_cache'):
            self._noise_model_cache = {}
        
        # Create simple depolarizing noise directly
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        noise_model = NoiseModel()
        
        try:
            # Add depolarizing error to single-qubit gates
            error_1q = depolarizing_error(noise_level, 1)
            noise_model.add_all_qubit_quantum_error(error_1q, ['x', 'y', 'z'])
            
            # Add depolarizing error to two-qubit gates
            if noise_level > 0:
                error_2q = depolarizing_error(min(noise_level * 2, 0.99), 2)
                noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
                
            logger.info(f"Created simple noise model with level {noise_level}")
        except Exception as e:
            logger.warning(f"Noise model creation failed: {str(e)}, returning None")
            noise_model = None
        
        # Cache the noise model
        if noise_model is not None:
            self._noise_model_cache[cache_key] = noise_model
        
        return noise_model

    def _calculate_ideal_probabilities(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Calculate ideal probabilities for a circuit without noise.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Dictionary of ideal probabilities
        """
        try:
            # Create a clean copy without measurements
            clean_circuit = circuit.copy()
            clean_circuit.remove_final_measurements(inplace=True)
            
            # Get statevector
            sv = Statevector.from_instruction(clean_circuit)
            statevector = sv.data
            
            # Calculate probabilities
            probs = np.abs(statevector)**2
            
            # Convert to dictionary format
            ideal_probs = {}
            num_qubits = clean_circuit.num_qubits
            for i, prob in enumerate(probs):
                if prob > 1e-10:  # Only include non-zero probabilities
                    bitstring = format(i, f'0{num_qubits}b')
                    ideal_probs[bitstring] = float(prob)
            
            return ideal_probs
            
        except Exception as e:
            logger.warning(f"Failed to calculate ideal probabilities: {str(e)}")
            return {}