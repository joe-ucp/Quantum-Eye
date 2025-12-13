import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

from .ucp import UCPIdentity
from .transform import UcpFrequencyTransform


class ResonanceDetector:
    """
    Computes similarity metrics between quantum state frequency signatures using normalized inner products.

    This class performs pattern matching by calculating the overlap between frequency-domain 
    representations of quantum states. Given an input state's frequency signature and a set of 
    reference signatures, it identifies the best match by computing normalized complex inner 
    products and returns confidence scores based on the overlap magnitude.

    The matching process:
    1. Takes frequency signatures (64x64 complex arrays from the frequency transform)
    2. Computes normalized overlap scores between input and each reference
    3. Returns detection results with confidence metrics and best match identification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the resonance detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_threshold = self.config.get('detection_threshold', 0.7)
        self.metrics_extractor = UCPIdentity()
        self.transformer = UcpFrequencyTransform()
        
        # Default transform parameters
        self.alpha = self.config.get('alpha', 0.5)
        self.beta = self.config.get('beta', 0.5)
    
    def detect_resonance(self, reference_states: Dict[str, np.ndarray], 
                        input_state: np.ndarray, 
                        threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect resonance between input state and reference states.
        
        Args:
            reference_states: Dictionary of reference states or their frequency signatures
            input_state: Input quantum state to analyze
            threshold: Detection threshold (optional)
            
        Returns:
            Dictionary with detection results
        """
        # Get configuration settings
        detection_threshold = threshold or self.default_threshold
        
        # Extract UCP identity from input state
        input_identity = self.metrics_extractor.phi(input_state)

        
        # Transform to frequency domain with configured parameters
        input_signature = self.transformer.transform(
            input_identity, alpha=self.alpha, beta=self.beta)
        
        # Process each reference state and calculate overlap
        resonance_results = {}
        best_match = None
        best_match_overlap = 0.0
        
        for name, reference in reference_states.items():
            # Get reference signature 
            ref_signature = self._get_reference_signature(reference)
            
            # Calculate overlap between signatures
            overlap = self.frequency_signature_overlap(
                input_signature, ref_signature)
            
            # Calculate confidence based on overlap
            overlap_magnitude = abs(overlap)
            confidence = self._calculate_confidence(overlap_magnitude, detection_threshold)
            
            # Store result
            resonance_results[name] = {
                'overlap': float(overlap_magnitude),
                'confidence': float(confidence),
                'detected': overlap_magnitude > detection_threshold
            }
            
            # Track best match
            if overlap_magnitude > best_match_overlap:
                best_match = name
                best_match_overlap = overlap_magnitude
        
        # Return comprehensive results
        return {
            "input_state": input_state,
            "ucp_identity": input_identity,
            "frequency_signature": input_signature,
            "resonance_results": resonance_results,
            "best_match": best_match,
            "best_match_overlap": float(best_match_overlap),
            "detection_successful": best_match_overlap > detection_threshold,
            "confidence": float(self._calculate_confidence(best_match_overlap, detection_threshold)),
            "threshold": detection_threshold,
            "parameters": {
                "alpha": self.alpha,
                "beta": self.beta
            }
        }
    
    def _get_reference_signature(self, reference):
        """Get frequency signature from reference state or use existing signature."""
        if isinstance(reference, dict) and 'full_transform' in reference:
            return reference
        else:
            identity = self.metrics_extractor.phi(reference)
            return self.transformer.transform(identity, alpha=self.alpha, beta=self.beta)
    
    def frequency_signature_overlap(self, sig1, sig2):
        """
        Calculate overlap between two frequency signatures.
        
        Args:
            sig1: First frequency signature
            sig2: Second frequency signature
            
        Returns:
            Complex overlap value
        """
        # Extract full transforms
        transform1 = sig1['full_transform']
        transform2 = sig2['full_transform']
        
        # Ensure compatible shapes
        if transform1.shape != transform2.shape:
            # Resize to match if necessary
            min_shape = (min(transform1.shape[0], transform2.shape[0]),
                         min(transform1.shape[1], transform2.shape[1]))
            transform1 = transform1[:min_shape[0], :min_shape[1]]
            transform2 = transform2[:min_shape[0], :min_shape[1]]
        
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
        
        return overlap
    
    def _calculate_confidence(self, overlap, threshold):
        """
        Convert overlap to confidence score.
        
        Args:
            overlap: Overlap magnitude
            threshold: Detection threshold
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if overlap < threshold:
            # Below threshold: linear scale from 0 to 0.5
            return 0.5 * (overlap / threshold)
        else:
            # Above threshold: scale from 0.5 to 1.0
            remaining_range = 1.0 - threshold
            if remaining_range > 0:
                above_threshold = overlap - threshold
                return 0.5 + 0.5 * (above_threshold / remaining_range)
            else:
                return 1.0  # Avoid division by zero