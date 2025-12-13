"""
Frequency domain transformation module for the Quantum Eye framework.

This module implements specialized 2D Fourier transforms that map quantum state 
features to frequency space where signal and noise characteristics naturally separate.
"""

import numpy as np
from scipy.ndimage import zoom

class UcpFrequencyTransform:
    """
    Transforms quantum state features to frequency domain for pattern analysis.
    
    This class implements a modified 2D Fourier transform that creates 64x64 complex
    frequency signatures from quantum state features. The transformation enhances 
    pattern recognition by separating signal and noise into different frequency bands,
    enabling robust state identification even with noisy measurements.
    """
    
    def __init__(self, config=None):
        """
        Initialize the frequency transformer.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - resolution: Grid resolution for frequency transforms (default: 64)
                - min_component_weight: Minimum weight for feature components (default: 0.15)
                - enable_entanglement_emphasis: Whether to apply special patterns for
                  entangled states (default: True)
        """
        self.config = config or {}
        self.resolution = self.config.get('resolution', 64)  # Resolution of frequency grid
        self.min_component_weight = self.config.get('min_component_weight', 0.15)
        self.enable_entanglement_emphasis = self.config.get('enable_entanglement_emphasis', True)
    
    def transform(self, ucp_identity, alpha=0.5, beta=0.5):
        """
        Transform quantum state features to frequency domain representation.
        
        Creates a 64x64 complex frequency signature by arranging input features in a 
        2D pattern and applying a specialized Fourier transform with adaptive parameters.
        
        Args:
            ucp_identity: Dictionary containing extracted quantum state features
            alpha: Parameter controlling frequency band emphasis (0-1)
            beta: Parameter controlling feature mixing ratios (0-1)
            
        Returns:
            Dictionary with frequency domain components and full 64x64 transform
        """
        # Validate input parameters
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0, beta))
        
        # Handle different input types
        if isinstance(ucp_identity, np.ndarray):
            return self._transform_array_signature(ucp_identity, alpha, beta)
            
        # Check if ucp_identity is valid
        if not isinstance(ucp_identity, dict):
            return {'full_transform': np.zeros((self.resolution, self.resolution), dtype=complex)}
        
        # Extract UCP components
        phase_coherence = ucp_identity.get("phase_coherence_metrics")
        state_distribution = ucp_identity.get("state_distribution_metrics")
        entropic_measures = ucp_identity.get("entropic_measures_metrics")
        quantum_correlations = ucp_identity.get("entanglement_metrics")
        
        # Initialize frequency components
        frequency_components = {
            'phase_coherence_metrics': None,
            'state_distribution_metrics': None,
            'entropic_measures_metrics': None,
            'quantum_correlations': None,
            'full_transform': None
        }
        
        # Transform each component with adaptive parameters
        if phase_coherence is not None:
            frequency_components['phase_coherence_metrics'] = self.map_phase_coherence(
                phase_coherence, alpha, beta)
        
        if state_distribution is not None:
            frequency_components['state_distribution_metrics'] = self.map_state_distribution(
                state_distribution, alpha, beta)
        
        if entropic_measures is not None:
            frequency_components['entropic_measures_metrics'] = self.map_entropic_measures(
                entropic_measures, alpha, beta)
        
        if quantum_correlations is not None:
            frequency_components['quantum_correlations'] = self.map_quantum_correlations(
                quantum_correlations, alpha, beta)
        
        # Get component weights with minimum value to ensure all components contribute
        p_weight = max(phase_coherence.get("weight", 1.0), self.min_component_weight) if phase_coherence else self.min_component_weight
        s_weight = max(state_distribution.get("weight", 1.0), self.min_component_weight) if state_distribution else self.min_component_weight
        e_weight = max(entropic_measures.get("weight", 1.0), self.min_component_weight) if entropic_measures else self.min_component_weight
        q_weight = max(quantum_correlations.get("weight", 1.0), self.min_component_weight) if quantum_correlations else self.min_component_weight
        
        # Apply beta parameter to modify weight distribution
        # Beta controls the balance between components
        beta_factor = 1.0 + beta
        total_weight = p_weight + s_weight + e_weight + q_weight
        
        if total_weight > 0:
            # Adjust weights based on relative importance
            p_importance = phase_coherence.get("phase_coherence", 0.5) if phase_coherence else 0.5
            s_importance = state_distribution.get("ipr_metric", 0.5) if state_distribution else 0.5
            e_importance = entropic_measures.get("normalized_entropy", 0.5) if entropic_measures else 0.5
            q_importance = quantum_correlations.get("entanglement_metric", 0.5) if quantum_correlations else 0.5
            
            # Calculate adaptive weights
            p_weight = p_weight * beta_factor * p_importance / total_weight
            s_weight = s_weight * beta_factor * s_importance / total_weight
            e_weight = e_weight * beta_factor * e_importance / total_weight
            q_weight = q_weight * beta_factor * q_importance / total_weight
            
            # Renormalize
            adj_total = p_weight + s_weight + e_weight + q_weight
            if adj_total > 0:
                p_weight /= adj_total
                s_weight /= adj_total
                e_weight /= adj_total
                q_weight /= adj_total
        
        # Initialize full transform array shape based on resolution
        full_transform = np.zeros((self.resolution, self.resolution), dtype=complex)
        component_count = 0
        
        # Add each component with proper weighting
        if frequency_components['phase_coherence_metrics'] is not None:
            full_transform += p_weight * frequency_components['phase_coherence_metrics']
            component_count += 1
        
        if frequency_components['state_distribution_metrics'] is not None:
            full_transform += s_weight * frequency_components['state_distribution_metrics']
            component_count += 1
        
        if frequency_components['entropic_measures_metrics'] is not None:
            full_transform += e_weight * frequency_components['entropic_measures_metrics']
            component_count += 1
        
        if frequency_components['quantum_correlations'] is not None:
            full_transform += q_weight * frequency_components['quantum_correlations']
            component_count += 1
        
        # Handle case when no components are available
        if component_count == 0:
            frequency_components['full_transform'] = full_transform
            return frequency_components
        
        # Apply additional emphasis for entangled states if enabled
        if self.enable_entanglement_emphasis and quantum_correlations is not None:
            quantum_correlations_factor = quantum_correlations.get("entanglement_metric", 0.0)
            if quantum_correlations_factor > 0.5:  # More emphasis for highly entangled states
                # Apply additional pattern to emphasize high quantum_correlations
                full_transform = self._apply_entanglement_emphasis(full_transform, quantum_correlations_factor)
        
        # Normalize if we have components
        norm = np.linalg.norm(full_transform)
        if norm > 0:
            full_transform = full_transform / norm
        
        # Check for NaN or inf values and replace them
        if np.any(np.isnan(full_transform)) or np.any(np.isinf(full_transform)):
            full_transform = np.nan_to_num(full_transform)
            # Renormalize after fixing NaN/inf values
            norm = np.linalg.norm(full_transform)
            if norm > 0:
                full_transform = full_transform / norm
        
        frequency_components['full_transform'] = full_transform
        
        return frequency_components
    
    def map_phase_coherence(self, phase_coherence, alpha=0.5, beta=0.5):
        """Map phase coherence component to frequency domain."""
        
        # Handle case where phase_coherence is a single float value
        if isinstance(phase_coherence, (int, float)):
            # Create a dictionary structure from the single value
            phase_coherence_dict = {
                "phase_coherence": float(phase_coherence),
                "phase_variance": 0.0,
                "phase_coherence_structure": 0.0,
                "phase_relationships": [],
                "superposition_degree": 1,
                "amplitude_entropy": 0.0,
                "global_phase": 0.0,
                "n_qubits": 1
            }
        else:
            phase_coherence_dict = phase_coherence
        
        # Extract key metrics with safety checks
        phase_coherence_val = phase_coherence_dict.get("phase_coherence", 0.0)
        phase_variance = phase_coherence_dict.get("phase_variance", 0.0)
        phase_coherence_structure = phase_coherence_dict.get("phase_coherence_structure", 0.0)
        phase_relationships = phase_coherence_dict.get("phase_relationships", [])
        
        # Extract additional metrics if available
        superposition_degree = phase_coherence_dict.get("superposition_degree", 1)
        amplitude_entropy = phase_coherence_dict.get("amplitude_entropy", 0.0)
        global_phase = phase_coherence_dict.get("global_phase", 0.0)
        n_qubits = phase_coherence_dict.get("n_qubits", 1)
        
        # Validate numeric values
        phase_coherence_val = float(phase_coherence_val)
        phase_variance = float(phase_variance)
        phase_coherence_structure = float(phase_coherence_structure)
        
        # Prepare encoding array
        encoding = np.zeros((self.resolution, self.resolution), dtype=complex)
        
        # Apply alpha parameter to control frequency distribution
        # Higher alpha emphasizes higher frequencies
        freq_factor = alpha * (1.0 - phase_coherence_val) + (1.0 - alpha) * phase_coherence_val
        
        # Use vectorized operations instead of nested loops
        i_coords, j_coords = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution), indexing='ij')
        x_norm = i_coords / self.resolution
        y_norm = j_coords / self.resolution
        
        # Calculate centered coordinates for radial patterns
        x_centered = x_norm - 0.5
        y_centered = y_norm - 0.5
        r = np.sqrt(x_centered**2 + y_centered**2) * 2  # Scale to [0,1] range
        theta = np.arctan2(y_centered, x_centered)
        
        # Different encoding patterns for different state types
        if superposition_degree <= 1:
            # Basis state pattern - simple phase gradient
            phase = 2 * np.pi * phase_coherence_val * (x_norm + y_norm) * freq_factor
            amplitude = 1.0 - (beta * phase_variance)
        else:
            # Superposition state patterns (differentiate between |+⟩ and Bell states)
            if n_qubits <= 1:
                # Single-qubit superposition (like |+⟩)
                pattern_scale = 3.0
                phase = 2 * np.pi * (x_norm + y_norm) * freq_factor
                # Add distinctive stripe pattern for |+⟩
                stripe_pattern = np.sin(pattern_scale * np.pi * x_norm)
                phase += stripe_pattern * np.pi * 0.25
                # Amplitude modulation
                amplitude = 1.0 - (beta * 0.5 * phase_variance)
                r_component = np.sin(2 * np.pi * r)
                amplitude = amplitude * (1.0 + 0.2 * r_component)
            else:
                # Multi-qubit superposition (like Bell states)
                pattern_scale = 4.0
                # Use phase_coherence structure to create different patterns
                theta_factor = phase_coherence_structure * 5.0
                # Create a pattern that mixes radial and angular components
                phase = 2 * np.pi * freq_factor * (
                    0.5 * (x_norm + y_norm) + 
                    0.5 * (r * np.cos(theta * theta_factor))
                )
                # Amplitude modulation based on radial distance - distinctive for Bell states
                amplitude = 1.0 - (beta * 0.3 * phase_variance)
                r_mod = np.sin(pattern_scale * np.pi * r)
                amplitude = amplitude * (1.0 + 0.3 * r_mod)
        
        # Ensure non-zero amplitude
        amplitude = np.maximum(0.1, amplitude)
        
        # Complex encoding with global phase offset
        encoding = amplitude * np.exp(1j * (phase + global_phase))
        
        # Encode phase relationships if available
        if phase_relationships and len(phase_relationships) > 0:
            # Scale relationships to fit the encoding
            scaled_relationships = np.array(phase_relationships) / np.pi
            scaled_relationships = scaled_relationships[:min(len(scaled_relationships), self.resolution)]
            
            # Create a pattern based on these relationships
            relationship_pattern = np.zeros(self.resolution, dtype=complex)
            for i, rel in enumerate(scaled_relationships):
                if i < self.resolution:
                    relationship_pattern[i] = np.exp(1j * rel * np.pi)
            
            # Apply pattern to encoding - vary by state type
            relationship_weight = 1.0
            if superposition_degree > 1:
                # Adjust relationship weight based on qubit count
                if n_qubits <= 1:
                    relationship_weight = 1.0 + 0.3 * amplitude_entropy  # |+⟩ state
                else:
                    relationship_weight = 1.0 + 0.6 * amplitude_entropy  # Bell state
                    
            for i in range(self.resolution):
                # Fixed check for empty phase_relationships
                if len(scaled_relationships) > 0:
                    pattern_idx = i % len(scaled_relationships)
                    encoding[i, :] *= relationship_pattern[pattern_idx] ** relationship_weight
        
        # Apply Fourier transform
        transformed = np.fft.fft2(encoding)
        
        # Apply phase_coherence structure modulation
        # Higher phase_coherence structure means more emphasis on lower frequencies
        if phase_coherence_structure > 0:
            structure_mask = self._generate_structure_mask(phase_coherence_structure)
            transformed = transformed * structure_mask
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def map_state_distribution(self, state_distribution, alpha=0.5, beta=0.5):
        """
        Map state_distribution closure component to frequency domain.
        
        Args:
            state_distribution: state distribution component from Quantum Signature Validation 
            alpha: Parameter controlling frequency band distribution
            beta: Parameter controlling operator balance
            
        Returns:
            Frequency domain representation of state_distribution
        """
        # Extract key metrics with safety checks
        ipr_metric = state_distribution.get("ipr_metric", 0.0)
        operator_balance = state_distribution.get("operator_balance", 0.0)
        operator_expectations = state_distribution.get("operator_expectations", {})
        
        # Validate numeric values
        ipr_metric = float(ipr_metric)
        operator_balance = float(operator_balance)
        
        # Convert operator expectations to array
        operator_array = self._build_operator_array(operator_expectations)
        
        # Apply beta parameter to modify operator balance
        op_balance_factor = beta + (1.0 - beta) * operator_balance
        operator_array = operator_array * op_balance_factor
        
        # Apply alpha parameter to adjust frequency content
        # Creates a frequency mask that scales with alpha
        freq_mask = self._generate_frequency_mask(alpha, center_offset=ipr_metric)
        
        # Ensure operator array matches resolution
        if operator_array.shape != (self.resolution, self.resolution):
            operator_array = self._resize_array(operator_array, (self.resolution, self.resolution))
        
        # Apply specialized Fourier-like transform
        transformed = np.fft.fft2(operator_array)
        
        # Apply frequency mask
        transformed = transformed * freq_mask
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def map_entropic_measures(self, entropic_measures, alpha=0.5, beta=0.5):
        """
        Map entropic_measures conservation component to frequency domain.
        
        Args:
            entropic_measures: entropic_measures conservation component from Quantum Signature Validation 
            alpha: Parameter controlling frequency distribution
            beta: Parameter controlling entropy emphasis
            
        Returns:
            Frequency domain representation of entropic_measures conservation
        """
        # Extract key metrics with safety checks
        entropy = entropic_measures.get("von_neumann_entropy", 0.0)
        normalized_entropy = entropic_measures.get("normalized_entropy", 0.0)
        entanglement_entropy = entropic_measures.get("entanglement_entropy", 0.0)
        
        # Validate numeric values
        entropy = float(entropy)
        normalized_entropy = float(normalized_entropy)
        entanglement_entropy = float(entanglement_entropy)
        
        # Create entropy encoding array using vectorized operations
        i_coords, j_coords = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution), indexing='ij')
        
        # Calculate radial distance from center
        x_centered = i_coords - self.resolution / 2
        y_centered = j_coords - self.resolution / 2
        r = np.sqrt(x_centered**2 + y_centered**2) / (self.resolution / 2)
        
        # Calculate angle
        theta = np.arctan2(y_centered, x_centered)
        
        # Apply alpha parameter for frequency distribution emphasis
        freq_phase = alpha * 2 * np.pi
        
        # Apply beta parameter for entropy emphasis
        entropy_weight = beta * normalized_entropy
        
        # Create distinctive patterns based on entropy and entanglement
        if normalized_entropy > 0.95 and entanglement_entropy > 0.9:
            # Highly entangled state with high entropy (Bell state)
            # Create distinctive spiral pattern
            phase = theta * 4 * entropy_weight + freq_phase * r
            amplitude = 1.0 - (r * normalized_entropy * 0.7)
        elif normalized_entropy > 0.95:
            # High entropy but not entangled (|+⟩ state)
            # Create distinctive linear pattern
            phase = freq_phase * (x_centered/self.resolution + y_centered/self.resolution) * 4
            amplitude = 1.0 - (r * normalized_entropy * 0.5)
        else:
            # Standard pattern for other states
            phase = theta * entropy_weight + freq_phase * r
            amplitude = 1.0 - (r * normalized_entropy)
        
        # Ensure non-zero amplitude
        amplitude = np.maximum(0.1, amplitude)
        
        # Create complex encoding
        encoding = amplitude * np.exp(1j * phase)
        
        # Apply transformation
        transformed = np.fft.fft2(encoding)
        
        # If entanglement entropy is significant, enhance specific frequencies
        if entanglement_entropy > 0.1:
            entanglement_mask = self._generate_entanglement_mask(entanglement_entropy)
            transformed = transformed * entanglement_mask
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def map_quantum_correlations(self, quantum_correlations, alpha=0.5, beta=0.5):
        """
        Map absolute quantum_correlations component to frequency domain.
        
        Args:
            quantum_correlations: quantum_correlations component from Quantum Signature Validation 
            alpha: Parameter controlling frequency distribution
            beta: Parameter controlling emphasis ratio
            
        Returns:
            Frequency domain representation of quantum_correlations
        """
        # Extract key metrics with safety checks
        entanglement_metric = quantum_correlations.get("entanglement_metric", 0.0)
        entanglement_spectrum = quantum_correlations.get("entanglement_spectrum", [])
        entanglement_uniformity = quantum_correlations.get("entanglement_uniformity", 0.0)
        
        # Validate numeric values
        entanglement_metric = float(entanglement_metric)
        entanglement_uniformity = float(entanglement_uniformity)
        
        # Create different patterns based on entanglement_metric value
        # This is critical for distinguishing |+⟩ (U=0.5) from Bell (U=1.0)
        if entanglement_metric > 0.8:  # Bell-like states - highly entangled
            return self._create_entanglement_pattern(entanglement_metric, alpha, beta)
        elif entanglement_metric > 0.4:  # Superposition but not entangled (|+⟩)
            return self._create_superposition_pattern(entanglement_metric, alpha, beta)
        else:  # Low quantum_correlations (basis states)
            return self._create_standard_pattern(entanglement_metric, alpha, beta)
    
    def _create_entanglement_pattern(self, quantum_correlations_val, alpha, beta):
        """
        Create a distinctive pattern for entangled states.
        
        Args:
            quantum_correlations_val: quantum_correlations signature value
            alpha: Alpha parameter
            beta: Beta parameter
            
        Returns:
            Frequency domain representation
        """
        # Implement a pattern with radial symmetry distinctive to entangled states
        i_coords, j_coords = np.meshgrid(
            np.arange(self.resolution), 
            np.arange(self.resolution), 
            indexing='ij'
        )
        
        # Create centered coordinates
        x_norm = (i_coords - self.resolution/2) / (self.resolution/2)
        y_norm = (j_coords - self.resolution/2) / (self.resolution/2)
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)
        
        # Scale alpha and beta based on quantum_correlations value
        alpha_scaled = alpha * (1.0 + quantum_correlations_val)
        beta_scaled = beta * (1.0 + quantum_correlations_val * 0.5)
        
        # Create radial pattern with high angular frequency (distinctive to entangled states)
        # Using high theta multiplier creates a distinctive spiral pattern
        phase = alpha_scaled * theta * 8 + beta_scaled * r * 2 * np.pi
        amplitude = np.cos(r * np.pi * 2) * 0.25 + 0.75
        
        # Create complex encoding
        encoding = amplitude * np.exp(1j * phase)
        
        # Apply a Fourier transform
        transformed = np.fft.fft2(encoding)
        
        # Apply additional emphasis for entangled states
        # Create concentric ring pattern in frequency domain
        center = self.resolution // 2
        freq_mask = np.ones((self.resolution, self.resolution), dtype=complex)
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                if i < self.resolution and j < self.resolution:
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    ring_factor = np.abs(np.sin(dist / 10))
                    freq_mask[i, j] = 1.0 + ring_factor * quantum_correlations_val
        
        transformed = transformed * freq_mask
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def _create_superposition_pattern(self, quantum_correlations_val, alpha, beta):
        """
        Create a distinctive pattern for superposition (non-entangled) states.
        
        Args:
            quantum_correlations_val: quantum_correlations signature value
            alpha: Alpha parameter
            beta: Beta parameter
            
        Returns:
            Frequency domain representation
        """
        # Implement a pattern with linear structure distinctive to superposition states
        i_coords, j_coords = np.meshgrid(
            np.arange(self.resolution), 
            np.arange(self.resolution), 
            indexing='ij'
        )
        
        # Create normalized coordinates
        x_norm = i_coords / self.resolution
        y_norm = j_coords / self.resolution
        
        # Scale parameters
        alpha_scaled = alpha * (1.0 + quantum_correlations_val * 0.5)
        beta_scaled = beta * (1.0 + quantum_correlations_val * 0.3)
        
        # Create linear pattern with stripes (distinctive to |+⟩ state)
        # Using sin function creates a striped pattern
        stripe_freq = 5.0
        phase = 2 * np.pi * alpha_scaled * (x_norm + y_norm) + beta_scaled * np.sin(stripe_freq * np.pi * x_norm)
        amplitude = 1.0 - 0.3 * np.sin(stripe_freq * np.pi * y_norm)**2
        
        # Create complex encoding
        encoding = amplitude * np.exp(1j * phase)
        
        # Apply a Fourier transform
        transformed = np.fft.fft2(encoding)
        
        # Apply grid pattern in frequency domain - distinctive for |+⟩
        freq_mask = np.ones((self.resolution, self.resolution), dtype=complex)
        
        # Create grid pattern
        grid_size = max(2, int(8 * quantum_correlations_val))
        for i in range(self.resolution):
            for j in range(self.resolution):
                if (i % grid_size < grid_size//2) and (j % grid_size < grid_size//2):
                    freq_mask[i, j] = 1.0 + 0.5 * quantum_correlations_val
        
        transformed = transformed * freq_mask
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def _create_standard_pattern(self, quantum_correlations_val, alpha, beta):
        """
        Create a standard pattern for basis states or low-quantum_correlations states.
        
        Args:
            quantum_correlations_val: quantum_correlations signature value
            alpha: Alpha parameter
            beta: Beta parameter
            
        Returns:
            Frequency domain representation
        """
        # Create quantum_correlations encoding using vectorized operations
        i_coords, j_coords = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution), indexing='ij')
        
        # Scale alpha based on quantum_correlations signature
        alpha_scaled = alpha * (1.0 + quantum_correlations_val)
        
        # Scale beta based on quantum_correlations value
        beta_scaled = beta * (1.0 + quantum_correlations_val)
        
        # Calculate pattern coordinates
        angle = 2 * np.pi * (i_coords / self.resolution) * alpha_scaled
        radius = (j_coords / self.resolution) * beta_scaled
        
        # Create complex values with uniform magnitude but varying phase
        magnitude = np.ones_like(angle)
        phase = angle + radius * 2 * np.pi * quantum_correlations_val
        
        encoding = magnitude * np.exp(1j * phase)
        
        # Apply specialized transform
        transformed = np.fft.fft2(encoding)
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
            
        return transformed
    
    def _apply_entanglement_emphasis(self, transform, quantum_correlations_factor):
        """
        Apply special emphasis pattern to highlight entangled states.
        
        This function adds a distinctive pattern to emphasize entangled states,
        helping differentiate Bell states from other superposition states.
        
        Args:
            transform: Original frequency transform
            quantum_correlations_factor: quantum_correlations signature value (entanglement measure)
            
        Returns:
            Modified transform with entanglement emphasis
        """
        height, width = transform.shape
        i_coords, j_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate coordinates relative to center
        x_norm = (i_coords - width/2) / (width/2)
        y_norm = (j_coords - height/2) / (height/2)
        
        # Create a distinctive frequency pattern only for entangled states
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)
        
        # Pattern modulation based on entanglement
        # This creates a unique spiral pattern that will only be prominent in entangled states
        spiral_freq = 8.0 * quantum_correlations_factor
        ring_freq = 3.0 * quantum_correlations_factor
        
        pattern = np.exp(1j * theta * spiral_freq) * np.exp(-r * 3) * np.cos(r * ring_freq * np.pi)**2
        
        # Normalize pattern
        pattern = pattern / np.linalg.norm(pattern)
        
        # Mix with original transform - higher mixing for higher quantum_correlations values
        mixture = transform + pattern * quantum_correlations_factor * 0.6
        
        # Normalize
        return mixture / np.linalg.norm(mixture)
    
    def _transform_array_signature(self, array_signature, alpha=0.5, beta=0.5):
        """
        Transform raw array signature to frequency domain.
        
        This is used when the input is already a NumPy array rather than a QSV identity dict.
        
        Args:
            array_signature: Input array
            alpha: Adaptive parameter controlling frequency band distribution
            beta: Adaptive parameter controlling component mixing
            
        Returns:
            Dictionary with frequency transform
        """
        # Handle non-array inputs
        try:
            array_signature = np.array(array_signature, dtype=complex)
        except:
            return {'full_transform': np.zeros((self.resolution, self.resolution), dtype=complex)}
            
        # Check for NaN or inf values
        if np.any(np.isnan(array_signature)) or np.any(np.isinf(array_signature)):
            array_signature = np.nan_to_num(array_signature)
        
        # Resize array to match resolution if needed
        if array_signature.shape != (self.resolution, self.resolution):
            # Use resize or interpolate to match resolution
            resized = self._resize_array(array_signature, (self.resolution, self.resolution))
        else:
            resized = array_signature.copy()
        
        # Apply frequency transform
        transformed = np.fft.fft2(resized)
        
        # Apply alpha parameter (frequency band emphasis)
        if alpha != 0.5:
            transformed = self._apply_alpha_emphasis(transformed, alpha)
        
        # Apply beta parameter (component emphasis)
        if beta != 0.5:
            transformed = self._apply_beta_emphasis(transformed, beta)
        
        # Normalize
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed = transformed / norm
        
        return {'full_transform': transformed}
    
    def _resize_array(self, array, new_shape):
        """
        Resize array to new shape using SciPy's zoom for better accuracy.
        
        Args:
            array: Input array
            new_shape: Target shape
            
        Returns:
            Resized array
        """
        # Calculate zoom factors
        if array.shape[0] == 0 or array.shape[1] == 0:
            return np.zeros(new_shape, dtype=complex)
            
        zoom_factors = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])
        
        # Handle complex arrays by processing real and imaginary parts separately
        real_part = zoom(np.real(array), zoom_factors, order=1)
        imag_part = zoom(np.imag(array), zoom_factors, order=1)
        
        # Combine back to complex array
        return real_part + 1j * imag_part
    
    def _apply_alpha_emphasis(self, transformed, alpha):
        """
        Apply alpha parameter emphasis to transformed array.
        
        Alpha controls frequency band distribution:
        - alpha > 0.5: Emphasize higher frequencies
        - alpha < 0.5: Emphasize lower frequencies
        
        Args:
            transformed: Transformed array
            alpha: Alpha parameter (0-1)
            
        Returns:
            Modified transformed array
        """
        # Generate a frequency mask based on alpha
        height, width = transformed.shape
        y_freq, x_freq = np.meshgrid(
            np.fft.fftfreq(height, d=1.0),
            np.fft.fftfreq(width, d=1.0),
            indexing='ij'
        )
        
        # Calculate frequency magnitude at each point
        freq_magnitude = np.sqrt(x_freq**2 + y_freq**2)
        max_magnitude = np.max(freq_magnitude)
        
        # Prevent division by zero
        if max_magnitude > 0:
            norm_magnitude = freq_magnitude / max_magnitude
        else:
            norm_magnitude = freq_magnitude
        
        # Create mask that emphasizes frequencies based on alpha
        if alpha > 0.5:
            # Emphasis increases with frequency
            emphasis = 1.0 + (alpha - 0.5) * 2.0 * norm_magnitude
        else:
            # Emphasis decreases with frequency
            emphasis = 1.0 + (0.5 - alpha) * 2.0 * (1.0 - norm_magnitude)
        
        # Apply emphasis
        return transformed * emphasis
    
    def _apply_beta_emphasis(self, transformed, beta):
        """
        Apply beta parameter emphasis to transformed array.
        
        Beta controls phase-amplitude relationship:
        - beta > 0.5: Emphasize phase entropic_measures
        - beta < 0.5: Emphasize amplitude entropic_measures
        
        Args:
            transformed: Transformed array
            beta: Beta parameter (0-1)
            
        Returns:
            Modified transformed array
        """
        # Split into amplitude and phase
        amplitude = np.abs(transformed)
        phase = np.angle(transformed)
        
        # Modify amplitude-phase relationship based on beta
        if beta > 0.5:
            # Emphasize phase (reduce amplitude variation)
            amplitude_mod = amplitude ** (1.0 - (beta - 0.5) * 2.0)
        else:
            # Emphasize amplitude (enhance amplitude variation)
            amplitude_mod = amplitude ** (1.0 + (0.5 - beta) * 2.0)
        
        # Normalize modified amplitude
        max_amplitude = np.max(amplitude_mod)
        if max_amplitude > 0:
            amplitude_mod = amplitude_mod / max_amplitude
        
        # Recombine amplitude and phase
        return amplitude_mod * np.exp(1j * phase)
    
    def _generate_structure_mask(self, phase_coherence_structure):
        """
        Generate a frequency mask based on phase_coherence structure.
        
        Args:
            phase_coherence_structure: phase_coherence structure metric (0-1)
            
        Returns:
            Frequency mask array
        """
        # Create frequency grid
        y_freq, x_freq = np.meshgrid(
            np.fft.fftfreq(self.resolution, d=1.0),
            np.fft.fftfreq(self.resolution, d=1.0),
            indexing='ij'
        )
        
        # Calculate distance from center
        dist_from_center = np.sqrt(x_freq**2 + y_freq**2)
        
        # Normalize distances
        max_dist = np.max(dist_from_center)
        if max_dist > 0:
            norm_dist = dist_from_center / max_dist
        else:
            norm_dist = dist_from_center
        
        # Create mask based on phase_coherence structure
        # Higher structure means more emphasis on lower frequencies
        mask = 1.0 + phase_coherence_structure * (1.0 - norm_dist)
        
        return mask
    
    def _generate_frequency_mask(self, alpha, center_offset=0.0):
        """
        Generate a frequency mask based on alpha parameter.
        
        Args:
            alpha: Parameter controlling frequency band distribution
            center_offset: Offset for the center of the mask
            
        Returns:
            Frequency mask array
        """
        # Create frequency grid
        y_freq, x_freq = np.meshgrid(
            np.fft.fftfreq(self.resolution, d=1.0),
            np.fft.fftfreq(self.resolution, d=1.0),
            indexing='ij'
        )
        
        # Calculate distance from center (with offset)
        center_x = 0.0 + center_offset
        center_y = 0.0 + center_offset
        dist_from_center = np.sqrt((x_freq - center_x)**2 + (y_freq - center_y)**2)
        
        # Normalize distances
        max_dist = np.max(dist_from_center)
        if max_dist > 0:
            norm_dist = dist_from_center / max_dist
        else:
            norm_dist = dist_from_center
        
        # Create mask based on alpha
        if alpha > 0.5:
            # Emphasize higher frequencies
            mask = 1.0 + (alpha - 0.5) * 2.0 * norm_dist
        else:
            # Emphasize lower frequencies
            mask = 1.0 + (0.5 - alpha) * 2.0 * (1.0 - norm_dist)
        
        return mask
    
    def _generate_entanglement_mask(self, entanglement_entropy):
        """
        Generate a mask that emphasizes frequencies associated with entanglement.
        
        Args:
            entanglement_entropy: Entanglement entropy measure
            
        Returns:
            Frequency mask array
        """
        # Create frequency grid
        y_freq, x_freq = np.meshgrid(
            np.fft.fftfreq(self.resolution, d=1.0),
            np.fft.fftfreq(self.resolution, d=1.0),
            indexing='ij'
        )
        
        # Calculate ring distance (distance from unit circle)
        ring_distance = np.abs(np.sqrt(x_freq**2 + y_freq**2) - 0.25)
        
        # Create ring emphasis based on entanglement_entropy
        emphasis = 1.0 + 5.0 * entanglement_entropy * np.exp(-10.0 * ring_distance)
        
        return emphasis
    
    def _build_operator_array(self, operator_expectations):
        """
        Build a 2D array from operator expectations.
        
        Args:
            operator_expectations: Dictionary of operator expectation values
            
        Returns:
            2D array encoding operator expectations
        """
        # Create an empty array
        operator_array = np.zeros((self.resolution, self.resolution), dtype=complex)
        
        # If no operator expectations, return empty array
        if not operator_expectations:
            return operator_array
        
        # Extract X, Y, Z expectations for each qubit
        x_expectations = []
        y_expectations = []
        z_expectations = []
        
        for key, value in operator_expectations.items():
            if key.startswith('X_'):
                try:
                    qubit_idx = int(key.split('_')[1])
                    x_expectations.append((qubit_idx, value))
                except (ValueError, IndexError):
                    continue
            elif key.startswith('Y_'):
                try:
                    qubit_idx = int(key.split('_')[1])
                    y_expectations.append((qubit_idx, value))
                except (ValueError, IndexError):
                    continue
            elif key.startswith('Z_'):
                try:
                    qubit_idx = int(key.split('_')[1])
                    z_expectations.append((qubit_idx, value))
                except (ValueError, IndexError):
                    continue
        
        # Sort by qubit index
        x_expectations.sort(key=lambda x: x[0])
        y_expectations.sort(key=lambda x: x[0])
        z_expectations.sort(key=lambda x: x[0])
        
        # Extract values only
        x_values = [x[1] for x in x_expectations]
        y_values = [y[1] for y in y_expectations]
        z_values = [z[1] for z in z_expectations]
        
        # Determine number of qubits
        n_qubits = max(len(x_values), len(y_values), len(z_values))
        if n_qubits == 0:
            return operator_array
        
        # Pad lists to ensure they're all the same length
        while len(x_values) < n_qubits:
            x_values.append(0.0)
        while len(y_values) < n_qubits:
            y_values.append(0.0)
        while len(z_values) < n_qubits:
            z_values.append(0.0)
        
        # Create grid sections for X, Y, Z values
        grid_size = max(1, int(np.ceil(np.sqrt(n_qubits))))
        x_grid = np.zeros((grid_size, grid_size))
        y_grid = np.zeros((grid_size, grid_size))
        z_grid = np.zeros((grid_size, grid_size))
        
        # Fill in the grids with bounds checking
        for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:  # Ensure we don't exceed grid bounds
                x_grid[row, col] = x
                y_grid[row, col] = y
                z_grid[row, col] = z
        
        # Place the grids in the operator array
        # Divide the array into 3 sections for X, Y, Z
        h_third = max(1, self.resolution // 3)
        
        # Resize grids to fit their sections
        x_grid_resized = self._resize_array(x_grid, (h_third, self.resolution))
        y_grid_resized = self._resize_array(y_grid, (h_third, self.resolution))
        z_grid_resized = self._resize_array(z_grid, (h_third, self.resolution))
        
        # Place in the operator array with bounds checking
        operator_array[:h_third, :] = x_grid_resized
        operator_array[h_third:min(2*h_third, self.resolution), :] = y_grid_resized
        operator_array[min(2*h_third, self.resolution):min(3*h_third, self.resolution), :] = z_grid_resized
        
        return operator_array
        
    def frequency_signature_overlap(self, sig1, sig2):
        """
        Calculate normalized inner product between two frequency signatures.
        
        Computes the complex inner product between two 64x64 frequency arrays and 
        normalizes by their magnitudes, returning the absolute value. This provides 
        a similarity score between 0 (orthogonal/no match) and 1 (identical).
        
        The calculation: |⟨sig1|sig2⟩| / (||sig1|| × ||sig2||)
        
        Args:
            sig1: First frequency signature dictionary containing 'full_transform' key
            sig2: Second frequency signature dictionary containing 'full_transform' key
            
        Returns:
            float: Normalized overlap score between 0 and 1
        """
        # Extract full transforms
        transform1 = sig1.get('full_transform')
        transform2 = sig2.get('full_transform')
        
        # Handle None values
        if transform1 is None or transform2 is None:
            return 0.0
        
        # Calculate normalized inner product
        transform1_flat = transform1.flatten()
        transform2_flat = transform2.flatten()
        
        # Calculate norms
        norm1 = np.linalg.norm(transform1_flat)
        norm2 = np.linalg.norm(transform2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate normalized inner product (overlap)
        overlap = np.vdot(transform1_flat, transform2_flat) / (norm1 * norm2)
        
        return np.abs(overlap)
    
    def optimize_parameters(self, input_state, reference_states, initial_alpha=0.5, initial_beta=0.5, steps=5):
        """
        Find optimal transform parameters via grid search to maximize pattern matching accuracy.
        
        Performs a grid search over alpha and beta parameters to find the combination that 
        produces the highest overlap between the input state's frequency signature and any 
        of the reference signatures. This optimization improves pattern matching by tuning 
        the frequency transform to emphasize the most distinguishing features.
        
        The search tests parameter combinations:
        - alpha: Controls frequency band emphasis (low vs high frequencies)  
        - beta: Controls relative weighting between feature components
        
        Args:
            input_state: Quantum state or feature dictionary to optimize for
            reference_states: List of reference states/features to match against
            initial_alpha: Starting alpha value (0-1)
            initial_beta: Starting beta value (0-1)
            steps: Number of grid points to test in each parameter dimension
            
        Returns:
            Tuple[float, float, float]: (best_alpha, best_beta, best_overlap_score)
        """
        # Handle empty references
        if not reference_states:
            return initial_alpha, initial_beta, 0.5
            
        # Convert to list if single reference
        if not isinstance(reference_states, list):
            reference_states = [reference_states]
            
        # Initialize best parameters
        best_alpha = initial_alpha
        best_beta = initial_beta
        best_overlap = 0.0
        
        # Extract UCP identity from input state if needed
        if not isinstance(input_state, dict):
            from core.ucp import UCPIdentity
            metrics_extractor= UCPIdentity()
            input_ucp = metrics_extractor.phi(input_state)
        else:
            input_ucp = input_state
            
        # Process reference states
        processed_refs = []
        for ref in reference_states:
            if not isinstance(ref, dict):
                from core.ucp import UCPIdentity
                metrics_extractor= UCPIdentity()
                ref_ucp = metrics_extractor.phi(ref)
            else:
                ref_ucp = ref
            processed_refs.append(ref_ucp)
            
        # Define parameter search space
        alpha_range = np.linspace(0.2, 0.8, steps)
        beta_range = np.linspace(0.2, 0.8, steps)
        
        # Grid search for best parameters
        for alpha in alpha_range:
            for beta in beta_range:
                # Transform input state
                input_transform = self.transform(input_ucp, alpha=alpha, beta=beta)
                
                # Calculate overlap with each reference state
                max_overlap = 0.0
                for ref_ucp in processed_refs:
                    ref_transform = self.transform(ref_ucp, alpha=alpha, beta=beta)
                    overlap = self.frequency_signature_overlap(input_transform, ref_transform)
                    max_overlap = max(max_overlap, overlap)
                    
                # Update best parameters if improved
                if max_overlap > best_overlap:
                    best_overlap = max_overlap
                    best_alpha = alpha
                    best_beta = beta
                    
        return best_alpha, best_beta, best_overlap