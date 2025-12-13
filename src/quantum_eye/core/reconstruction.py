import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

from .ucp import UCPIdentity
from .transform import UcpFrequencyTransform


class BaseReconstructor:
    """
    Predicts quantum measurement outcomes from frequency-domain pattern signatures.
    
    This class implements the reconstruction phase of the Quantum Eye method, which uses 
    frequency signatures to predict measurement statistics in any basis. Rather than 
    recovering the exact quantum state, it identifies the reference state whose frequency 
    signature best matches the measured data and applies corrections based on the 
    confidence of the match.
    
    The reconstruction fidelity depends on the pattern matching confidence from the 
    frequency domain analysis, with phase corrections applied to align the predicted 
    measurements with observed data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize state reconstructor."""
        self.config = config or {}
        self.reference_states = {}
        
        # Initialize components
        self.metrics_extractor = UCPIdentity()
        self.transformer = UcpFrequencyTransform()
        
        # Get threshold values from config
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.9)
        self.medium_confidence_threshold = self.config.get('medium_confidence_threshold', 0.7)
    
    def register_reference_state(self, label: str, state: np.ndarray) -> bool:
        """
        Register a reference state for reconstruction.
        
        Args:
            label: Reference state label
            state: Reference quantum state
            
        Returns:
            Registration success
        """
        if state is None:
            return False
            
        self.reference_states[label] = state.copy()
        return True
    
    def reconstruct(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct an ideal quantum state from detection results.
        
        Args:
            detection_result: Result from resonance detection
            
        Returns:
            Dictionary with reconstruction results
        """
        # Extract information from detection result
        best_match = detection_result.get("best_match")
        confidence = detection_result.get("best_match_overlap", 0.0)
        input_state = detection_result.get("input_state")
        
        # Ensure we have a valid best match
        if best_match is None or best_match not in self.reference_states:
            return self._handle_no_match(input_state)
        
        # Get reference state
        reference_state = self.reference_states[best_match]
        
        # Choose reconstruction method based on confidence
        if confidence > self.high_confidence_threshold:
            reconstructed, method = self._high_confidence_reconstruction(
                input_state, reference_state, detection_result)
                
        elif confidence > self.medium_confidence_threshold:
            reconstructed, method = self._medium_confidence_reconstruction(
                input_state, reference_state, detection_result)
                
        else:
            reconstructed, method = self._low_confidence_reconstruction(
                input_state, reference_state, detection_result)
        
        # Verify reconstruction quality
        verification_result = self.verify_reconstruction(
            input_state, reconstructed, reference_state)
        
        # Return complete reconstruction result
        return {
            "reconstructed_state": reconstructed,
            "reference_label": best_match,
            "reference_fidelity": verification_result["reference_fidelity"],
            "input_fidelity": verification_result["input_fidelity"],
            "confidence": float(confidence),
            "verified": verification_result["verified"],
            "method": method
        }
    
    def _handle_no_match(self, input_state: np.ndarray) -> Dict[str, Any]:
        """Handle case when no reference match is found."""
        return {
            "reconstructed_state": input_state,  # Return input as fallback
            "reference_label": None,
            "reference_fidelity": 0.0,
            "input_fidelity": 1.0,
            "confidence": 0.0,
            "verified": False,
            "method": "fallback"
        }
    
    def _high_confidence_reconstruction(self, input_state: np.ndarray, 
                                      reference_state: np.ndarray, 
                                      detection_result: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        Perform high confidence reconstruction.
        
        High confidence reconstruction primarily uses the reference state, 
        with phase correction based on the input state.
        
        Args:
            input_state: Input quantum state
            reference_state: Reference quantum state
            detection_result: Detection result
            
        Returns:
            Tuple of (reconstructed_state, method_name)
        """
        # Calculate phase correction
        phase_correction = self._calculate_phase_correction(
            input_state, reference_state)
        
        # Apply phase correction to reference state
        reconstructed = reference_state * np.exp(1j * phase_correction)
        
        # Ensure proper normalization
        reconstructed = self._normalize_state(reconstructed)
        
        return reconstructed, "high_confidence_phase_correction"
    
    def _medium_confidence_reconstruction(self, input_state: np.ndarray, 
                                        reference_state: np.ndarray, 
                                        detection_result: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        Perform medium confidence reconstruction.
        
        Medium confidence reconstruction blends the reference and input states
        based on the confidence level.
        
        Args:
            input_state: Input quantum state
            reference_state: Reference quantum state
            detection_result: Detection result
            
        Returns:
            Tuple of (reconstructed_state, method_name)
        """
        confidence = detection_result.get("best_match_overlap", 0.0)
        
        # Calculate blend ratio based on confidence
        # Scales from 0 at medium threshold to 1 at high threshold
        blend_ratio = (confidence - self.medium_confidence_threshold) / (
            self.high_confidence_threshold - self.medium_confidence_threshold)
        blend_ratio = max(0.0, min(1.0, blend_ratio))  # Clamp to [0,1]
        
        # Calculate phase correction
        phase_correction = self._calculate_phase_correction(
            input_state, reference_state)
        
        # Apply phase correction to reference state
        phase_corrected_ref = reference_state * np.exp(1j * phase_correction)
        
        # Blend corrected reference and input
        reconstructed = blend_ratio * phase_corrected_ref + (1.0 - blend_ratio) * input_state
        
        # Ensure proper normalization
        reconstructed = self._normalize_state(reconstructed)
        
        return reconstructed, f"medium_confidence_blend_{blend_ratio:.2f}"
    
    def _low_confidence_reconstruction(self, input_state: np.ndarray, 
                                     reference_state: np.ndarray, 
                                     detection_result: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        Perform low confidence reconstruction.
        
        Low confidence reconstruction applies simple noise filtering
        since we don't have high confidence in the match.
        
        Args:
            input_state: Input quantum state
            reference_state: Reference quantum state
            detection_result: Detection result
            
        Returns:
            Tuple of (reconstructed_state, method_name)
        """
        
        # Apply amplitude-based noise filtering
        reconstructed = self._apply_simple_noise_filter(
            input_state, reference_state, detection_result)
        
        # Ensure proper normalization
        reconstructed = self._normalize_state(reconstructed)
        
        return reconstructed, "low_confidence_filtering"
    
    def _apply_simple_noise_filter(self, input_state: np.ndarray, 
                                 reference_state: np.ndarray, 
                                 detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Apply a simple amplitude-based noise filter.
        
        Args:
            input_state: Input quantum state
            reference_state: Reference quantum state
            detection_result: Detection result
            
        Returns:
            Filtered state
        """
        # Get confidence
        confidence = detection_result.get("best_match_overlap", 0.0)
        
        # Calculate amplitude thresholds based on reference
        ref_amplitudes = np.abs(reference_state)
        max_amplitude = np.max(ref_amplitudes)
        threshold = max_amplitude * 0.1  # 10% of max amplitude
        
        # Apply threshold to input state
        filtered = input_state.copy()
        small_indices = np.abs(filtered) < threshold
        
        # Zero out small amplitudes (likely noise)
        filtered[small_indices] = 0.0
        
        # Normalize
        filtered = self._normalize_state(filtered)
        
        return filtered
    
    def _calculate_phase_correction(self, input_state: np.ndarray, 
                                  reference_state: np.ndarray) -> float:
        """
        Calculate optimal phase correction between states.
        
        Args:
            input_state: Input quantum state
            reference_state: Reference quantum state
            
        Returns:
            Phase correction angle
        """
        # Calculate inner product between states
        inner_product = np.vdot(reference_state, input_state)
        
        # Extract phase angle
        phase = np.angle(inner_product)
        
        return phase
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize a quantum state.
        
        Args:
            state: Quantum state to normalize
            
        Returns:
            Normalized quantum state
        """
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            return state / norm
        else:
            # If norm is too small, return zero state
            return np.zeros_like(state)
    
    def verify_reconstruction(self, input_state: np.ndarray, 
                            reconstructed_state: np.ndarray, 
                            reference_state: np.ndarray) -> Dict[str, Any]:
        """
        Verify the quality of reconstructed state.
        
        Args:
            input_state: Original input state
            reconstructed_state: Reconstructed state
            reference_state: Reference state
            
        Returns:
            Dictionary with verification results
        """
        # Calculate fidelity with input state
        input_fidelity = self._calculate_fidelity(reconstructed_state, input_state)
        
        # Calculate fidelity with reference state
        reference_fidelity = self._calculate_fidelity(reconstructed_state, reference_state)
        
        # Calculate input-reference fidelity for comparison
        input_reference_fidelity = self._calculate_fidelity(input_state, reference_state)
        
        # Calculate effective improvement
        if input_reference_fidelity < 1.0:
            improvement_factor = (reference_fidelity - input_reference_fidelity) / (1.0 - input_reference_fidelity)
        else:
            improvement_factor = 0.0
        
        # Determine if verified (meets quality standards)
        verification_threshold = self.config.get('verification_threshold', 0.9)
        verified = (reference_fidelity > verification_threshold and 
                   improvement_factor > 0.0)
        
        return {
            "verified": verified,
            "input_fidelity": float(input_fidelity),
            "reference_fidelity": float(reference_fidelity),
            "input_reference_fidelity": float(input_reference_fidelity),
            "improvement_factor": float(improvement_factor)
        }
    
    def _calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity (0-1)
        """
        # Calculate inner product
        inner_product = np.vdot(state1, state2)
        
        # Fidelity is the squared absolute value of the inner product
        fidelity = np.abs(inner_product)**2
        
        return float(fidelity)


class BellStateReconstructor(BaseReconstructor):
    """
    Specialized reconstruction for 2-qubit entangled states with Bell state enhancement.
    
    Extends the base reconstruction with Bell-state-specific optimizations:
    - Pre-computed reference library of all four Bell states (Φ+, Φ-, Ψ+, Ψ-)
    - Entanglement enhancement that blends reconstructed states with ideal Bell states
      when concurrence falls below 0.95
    - Specialized fidelity calculations using concurrence as the entanglement metric
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Bell state reconstructor."""
        super().__init__(config)
        
        # Create standard Bell states as references
        self._create_standard_bell_states()
    
    def _create_standard_bell_states(self):
        """Create standard Bell states as references."""
        # Bell Φ+ = (|00⟩ + |11⟩)/√2
        bell_phi_plus = np.zeros(4, dtype=complex)
        bell_phi_plus[0] = 1/np.sqrt(2)  # |00⟩
        bell_phi_plus[3] = 1/np.sqrt(2)  # |11⟩
        self.register_reference_state("bell_phi_plus", bell_phi_plus)
        
        # Bell Φ- = (|00⟩ - |11⟩)/√2
        bell_phi_minus = np.zeros(4, dtype=complex)
        bell_phi_minus[0] = 1/np.sqrt(2)   # |00⟩
        bell_phi_minus[3] = -1/np.sqrt(2)  # -|11⟩
        self.register_reference_state("bell_phi_minus", bell_phi_minus)
        
        # Bell Ψ+ = (|01⟩ + |10⟩)/√2
        bell_psi_plus = np.zeros(4, dtype=complex)
        bell_psi_plus[1] = 1/np.sqrt(2)  # |01⟩
        bell_psi_plus[2] = 1/np.sqrt(2)  # |10⟩
        self.register_reference_state("bell_psi_plus", bell_psi_plus)
        
        # Bell Ψ- = (|01⟩ - |10⟩)/√2
        bell_psi_minus = np.zeros(4, dtype=complex)
        bell_psi_minus[1] = 1/np.sqrt(2)   # |01⟩
        bell_psi_minus[2] = -1/np.sqrt(2)  # -|10⟩
        self.register_reference_state("bell_psi_minus", bell_psi_minus)
    
    def _high_confidence_reconstruction(self, input_state: np.ndarray, 
                                      reference_state: np.ndarray, 
                                      detection_result: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        High-confidence reconstruction with entanglement enhancement for Bell states.
        
        Applies standard phase correction then optionally enhances entanglement by blending
        with the nearest ideal Bell state if concurrence < 0.95. This ensures reconstructed
        states maintain strong entanglement properties characteristic of Bell states.
        
        Args:
            input_state: Noisy measurement data
            reference_state: Best-matching reference from frequency analysis  
            detection_result: Pattern matching results including confidence scores
            
        Returns:
            Tuple of (enhanced_state, method_description)
        """
        # First apply standard reconstruction
        reconstructed, method = super()._high_confidence_reconstruction(
            input_state, reference_state, detection_result)
        
        # Check if we need to enhance Bell state properties
        if self.config.get('enhance_entanglement', True):
            # Calculate concurrence as entanglement measure
            concurrence = self._calculate_concurrence(reconstructed)
            
            # If entanglement is too low, enhance it
            if concurrence < 0.95:
                # Get closest perfect Bell state
                bell_state, bell_label = self._find_closest_bell_state(reconstructed)
                
                # Blend with perfect Bell state to enhance entanglement
                enhance_ratio = 0.3
                enhanced = (1.0 - enhance_ratio) * reconstructed + enhance_ratio * bell_state
                
                # Normalize
                enhanced = self._normalize_state(enhanced)
                
                # Check if enhancement worked
                new_concurrence = self._calculate_concurrence(enhanced)
                
                if new_concurrence > concurrence:
                    reconstructed = enhanced
                    method = f"{method}_enhanced_{bell_label}"
        
        return reconstructed, method
    
    def _calculate_concurrence(self, state: np.ndarray) -> float:
        """
        Calculate concurrence entanglement measure for two-qubit states.
        
        Implements the Wootters concurrence formula: C = max(0, √λ₁ - √λ₂ - √λ₃ - √λ₄)
        where λᵢ are eigenvalues of ρ·(σy⊗σy)·ρ*·(σy⊗σy) in descending order.
        
        Args:
            state: Two-qubit quantum state vector
            
        Returns:
            Concurrence value between 0 (separable) and 1 (maximally entangled)
        """
        # Ensure state is a 2-qubit state
        if len(state) != 4:
            return 0.0
        
        # Reshape to 2x2 matrix form for easier calculation
        psi = state.reshape(2, 2)
        
        # Calculate spin-flipped state
        sigma_y = np.array([[0, -1j], [1j, 0]])
        psi_tilde = np.matmul(sigma_y, np.matmul(psi.conj(), sigma_y))
        
        # Calculate R matrix
        r_matrix = np.matmul(psi, psi_tilde)
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(r_matrix)
        
        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate concurrence
        concurrence = max(0, np.sqrt(eigenvalues[0]) - np.sqrt(eigenvalues[1]) - 
                         np.sqrt(eigenvalues[2]) - np.sqrt(eigenvalues[3]))
        
        return float(concurrence)
    
    def _find_closest_bell_state(self, state: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Find the closest standard Bell state.
        
        Args:
            state: Quantum state to analyze
            
        Returns:
            Tuple of (bell_state, bell_label)
        """
        best_match = None
        best_fidelity = 0.0
        
        for label, bell_state in self.reference_states.items():
            fidelity = self._calculate_fidelity(state, bell_state)
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_match = (bell_state, label)
        
        if best_match:
            return best_match
        else:
            # Fallback to Φ+ if no good match
            return (self.reference_states["bell_phi_plus"], "bell_phi_plus")