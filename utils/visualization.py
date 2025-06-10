import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.colors as colors

class VisualizationHelper:
    """
    BBasic visualization helpers for the Quantum Eye framework.

    Provides simple visualization functions for debugging and displaying
    the various components of the Quantum Eye framework including QSV 
    (Quantum Signature Validation) metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualization helper.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Set default figure size and style
        self.figsize = self.config.get('figsize', (10, 8))
        self.cmap = self.config.get('colormap', 'viridis')
        
        # Default color palette for components
        self.component_colors = {
            'phase_coherence_metrics': 'blue',
            'state_distribution_metrics': 'green',
            'entropic_measures_metrics': 'red',
            'entanglement_metrics': 'purple'
        }
    
    def plot_qsv_identity(self, qsv_identity: Dict[str, Any], title: str = "QSV Identity") -> plt.Figure:
        """
        Plot Quantum Signature Validation.
        
        Args:
            qsv_identity: QSV identity dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract key metrics
        if "quantum_signature" in qsv_identity:
            signature = qsv_identity["quantum_signature"]
            p_value = signature.get("P", 0)
            s_value = signature.get("S", 0)
            e_value = signature.get("E", 0)
            q_value = signature.get("Q", 0)
        else:
            p_value = 0
            s_value = 0
            e_value = 0
            q_value = 0
        
        # Calculate QSV score (P×S×E×Q > 0)
        qsv_score = p_value * s_value * e_value * q_value
        qsv_valid = qsv_score > 0
        
        # Plot component values
        labels = ['Phase Coherence (P)', 'State Distribution (S)', 'Entropic Measures (E)', 'Quantum Correlations (Q)']
        values = [p_value, s_value, e_value, q_value]
        colors = [self.component_colors['phase_coherence_metrics'], 
                self.component_colors['state_distribution_metrics'],
                self.component_colors['entropic_measures_metrics'],
                self.component_colors['entanglement_metrics']]
        
        # Bar chart of main components
        ax = axes[0, 0]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_title("QSV Component Values")
        
        # Radar chart
        ax = axes[0, 1]
        self._plot_radar(ax, labels, values, colors)
        ax.set_title("QSV Signature")
        
        # Component details - Phase Coherence
        ax = axes[1, 0]
        if "phase_coherence_metrics" in qsv_identity:
            logical = qsv_identity["phase_coherence_metrics"]
            logical_keys = ["phase_coherence", "logical_structure"]
            logical_values = [logical.get(key, 0) for key in logical_keys]
            ax.bar(logical_keys, logical_values, color=self.component_colors['phase_coherence_metrics'])
            ax.set_ylim(0, 1.0)
        ax.set_title("Phase Coherence Details")
        
        # Component details - Absolute Unity
        ax = axes[1, 1]
        if "entanglement_metrics" in qsv_identity:
            unity = qsv_identity["entanglement_metrics"]
            unity_keys = ["entanglement_metric", "normalized_entanglement"]
            unity_values = [unity.get(key, 0) for key in unity_keys]
            ax.bar(unity_keys, unity_values, color=self.component_colors['entanglement_metrics'])
            ax.set_ylim(0, 1.0)
        ax.set_title("Absolute Unity Details")
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_signature(self, frequency_signature: Dict[str, Any], 
                               title: str = "Frequency Signature") -> plt.Figure:
        """
        Plot frequency domain representation.
        
        Args:
            frequency_signature: Frequency signature dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Full transform (amplitude)
        ax = axes[0, 0]
        full_transform = frequency_signature.get('full_transform')
        if full_transform is not None:
            amplitude = np.abs(full_transform)
            im = ax.imshow(amplitude, cmap=self.cmap, interpolation='nearest')
            fig.colorbar(im, ax=ax)
        ax.set_title("Amplitude")
        
        # Full transform (phase)
        ax = axes[0, 1]
        if full_transform is not None:
            phase = np.angle(full_transform)
            im = ax.imshow(phase, cmap='hsv', interpolation='nearest')
            fig.colorbar(im, ax=ax)
        ax.set_title("Phase")
        
        # P and S components
        ax = axes[1, 0]
        p_transform = frequency_signature.get('phase_coherence_metrics')
        s_transform = frequency_signature.get('state_distribution_metrics')
        
        if p_transform is not None and s_transform is not None:
            # Average amplitude of P and S
            p_amplitude = np.abs(p_transform)
            s_amplitude = np.abs(p_transform)
            avg_lc = (p_amplitude + s_amplitude) / 2.0
            im = ax.imshow(avg_lc, cmap=self.cmap, interpolation='nearest')
            fig.colorbar(im, ax=ax)
        ax.set_title("P+S Components")
        
        # E and Q components
        ax = axes[1, 1]
        e_transform = frequency_signature.get('entropic_measures_metrics')
        q_transform = frequency_signature.get('quantum_correlations')
        
        if e_transform is not None and q_transform is not None:
            # Average amplitude of I and U
            e_amplitude = np.abs(e_transform)
            q_amplitude = np.abs(q_transform)
            avg_iu = (e_amplitude + q_amplitude) / 2.0
            im = ax.imshow(avg_iu, cmap=self.cmap, interpolation='nearest')
            fig.colorbar(im, ax=ax)
        ax.set_title("E+Q Components")
        
        plt.tight_layout()
        return fig
    
    def plot_resonance_results(self, detection_result: Dict[str, Any],
                             title: str = "Resonance Detection") -> plt.Figure:
        """
        Plot resonance detection results.
        
        Args:
            detection_result: Result from resonance detection
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        resonance_results = detection_result.get("resonance_results", {})
        if not resonance_results:
            # Create empty figure if no results
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No resonance results available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        names = []
        overlaps = []
        confidences = []
        detected = []
        
        for name, result in resonance_results.items():
            names.append(name)
            overlaps.append(result.get('overlap', 0))
            confidences.append(result.get('confidence', 0))
            detected.append(result.get('detected', False))
        
        # Overlaps bar chart
        ax = axes[0]
        bars = ax.bar(names, overlaps)
        
        # Color bars based on detection
        for i, bar in enumerate(bars):
            bar.set_color('green' if detected[i] else 'gray')
        
        # Add threshold line
        threshold = detection_result.get("threshold", 0.7)
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  label=f'Threshold: {threshold:.2f}')
        
        ax.set_ylim(0, 1.1)
        ax.set_title("Resonance Overlap")
        ax.legend()
        
        # Confidence bar chart
        ax = axes[1]
        bars = ax.bar(names, confidences)
        
        # Color bars based on detection
        for i, bar in enumerate(bars):
            bar.set_color('green' if detected[i] else 'gray')
        
        # Mark best match
        best_match = detection_result.get("best_match")
        if best_match:
            best_idx = names.index(best_match) if best_match in names else -1
            if best_idx >= 0:
                bars[best_idx].set_color('blue')
                ax.text(best_idx, confidences[best_idx] + 0.05, "Best Match", 
                       ha='center', fontsize=10)
        
        ax.set_ylim(0, 1.1)
        ax.set_title("Confidence Scores")
        
        plt.tight_layout()
        return fig
    
    def plot_reconstruction_results(self, reconstruction_result: Dict[str, Any],
                                  title: str = "State Reconstruction") -> plt.Figure:
        """
        Plot reconstruction results.
        
        Args:
            reconstruction_result: Result from state reconstruction
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        input_fidelity = reconstruction_result.get("input_fidelity", 0)
        reference_fidelity = reconstruction_result.get("reference_fidelity", 0)
        confidence = reconstruction_result.get("confidence", 0)
        method = reconstruction_result.get("method", "unknown")
        
        # Fidelity comparison
        ax = axes[0, 0]
        labels = ['Input Fidelity', 'Reference Fidelity']
        values = [input_fidelity, reference_fidelity]
        bars = ax.bar(labels, values)
        
        # Color coding
        bars[0].set_color('gray')
        bars[1].set_color('green')
        
        ax.set_ylim(0, 1.1)
        ax.set_title("Fidelity Comparison")
        
        # State visualization (if available)
        ax = axes[0, 1]
        
        # Try to visualize statevector amplitudes
        reconstructed_state = reconstruction_result.get("reconstructed_state")
        if reconstructed_state is not None:
            amplitudes = np.abs(reconstructed_state)
            
            # If too many elements, show only first 16
            if len(amplitudes) > 16:
                amplitudes = amplitudes[:16]
                basis_labels = [f"|{i:04b}⟩" for i in range(16)]
            else:
                n_qubits = int(np.log2(len(amplitudes)))
                basis_labels = [f"|{i:0{n_qubits}b}⟩" for i in range(len(amplitudes))]
            
            ax.bar(basis_labels, amplitudes)
            ax.set_title("Reconstructed State Amplitudes")
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "No state data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Reconstructed State")
        
        # Reconstruction method
        ax = axes[1, 0]
        # Display method name in text box
        ax.text(0.5, 0.5, f"Method: {method}", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        
        # Confidence level
        ax = axes[1, 1]
        self._plot_gauge(ax, confidence, "Confidence")
        
        plt.tight_layout()
        return fig
    
    def plot_full_pipeline(self, input_state: np.ndarray, 
                         detection_result: Dict[str, Any],
                         reconstruction_result: Dict[str, Any]) -> plt.Figure:
        """
        Plot full quantum eye pipeline overview.
        
        Args:
            input_state: Input quantum state
            detection_result: Detection results
            reconstruction_result: Reconstruction results
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Quantum Eye Pipeline Overview", fontsize=18)
        
        # Extract data
        qsv_identity = detection_result.get("qsv_identity")
        frequency_signature = detection_result.get("frequency_signature")
        best_match = detection_result.get("best_match", "None")
        best_overlap = detection_result.get("best_match_overlap", 0)
        reference_fidelity = reconstruction_result.get("reference_fidelity", 0)
        input_fidelity = reconstruction_result.get("input_fidelity", 0)
        
        # Input state probabilities
        ax = axes[0, 0]
        if input_state is not None:
            probabilities = np.abs(input_state)**2
            
            # If too many elements, show only first 16
            if len(probabilities) > 16:
                probabilities = probabilities[:16]
                basis_labels = [f"|{e:04b}⟩" for e in range(16)]
            else:
                n_qubits = int(np.log2(len(probabilities)))
                basis_labels = [f"|{e:0{n_qubits}b}⟩" for e in range(len(probabilities))]
            
            ax.bar(basis_labels, probabilities)
            ax.tick_params(axis='x', rotation=45)
        ax.set_title("Input State")
        
        # QSV Identity overview
        ax = axes[0, 1]
        if qsv_identity and "quantum_signature" in qsv_identity:
            signature = qsv_identity["quantum_signature"]
            labels = ['P', 'S', 'E', 'Q']
            values = [signature.get("P", 0), signature.get("S", 0), 
                     signature.get("E", 0), signature.get("Q", 0)]
            
            ax.bar(labels, values, color=[self.component_colors[f"{l.lower()}_consistency" 
                                                              if l == 'P' else 
                                                              f"{l.lower()}_conservation" 
                                                              if l == 'E' else
                                                              "entanglement_metrics" 
                                                              if l == 'Q' else
                                                              f"{l.lower()}_closure"] 
                                          for l in labels])
            ax.set_ylim(0, 1.0)
        else:
            ax.text(0.5, 0.5, "No QSV identity data", 
                   ha='center', va='center', fontsize=12)
        ax.set_title("QSV Identity")
        
        # Frequency domain visualization
        ax = axes[0, 2]
        if frequency_signature and 'full_transform' in frequency_signature:
            full_transform = frequency_signature['full_transform']
            amplitude = np.abs(full_transform)
            im = ax.imshow(amplitude, cmap=self.cmap, interpolation='nearest')
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "No frequency data", 
                   ha='center', va='center', fontsize=12)
        ax.set_title("Frequency Signature")
        
        # Detection results
        ax = axes[1, 0]
        if "resonance_results" in detection_result:
            results = detection_result["resonance_results"]
            names = list(results.keys())
            overlaps = [results[name].get('overlap', 0) for name in names]
            
            bars = ax.bar(names, overlaps)
            
            # Highlight best match
            if best_match in names:
                best_idx = names.index(best_match)
                bars[best_idx].set_color('blue')
            
            # Add threshold line
            threshold = detection_result.get("threshold", 0.7)
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      label=f'Threshold: {threshold:.2f}')
            
            ax.set_ylim(0, 1.1)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No detection results", 
                   ha='center', va='center', fontsize=12)
        ax.set_title("Detection Results")
        
        # Reconstruction information
        ax = axes[1, 1]
        reconstructed_state = reconstruction_result.get("reconstructed_state")
        if reconstructed_state is not None:
            probabilities = np.abs(reconstructed_state)**2
            
            # If too many elements, show only first 16
            if len(probabilities) > 16:
                probabilities = probabilities[:16]
                basis_labels = [f"|{i:04b}⟩" for i in range(16)]
            else:
                n_qubits = int(np.log2(len(probabilities)))
                basis_labels = [f"|{i:0{n_qubits}b}⟩" for i in range(len(probabilities))]
            
            ax.bar(basis_labels, probabilities)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "No reconstruction data", 
                   ha='center', va='center', fontsize=12)
        ax.set_title("Reconstructed State")
        
        # Metrics overview
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate QSV score (P×S×E×Q > 0) for metrics
        if qsv_identity and "quantum_signature" in qsv_identity:
            signature = qsv_identity["quantum_signature"]
            qsv_score = (signature.get("P", 0) * signature.get("S", 0) * 
                        signature.get("E", 0) * signature.get("Q", 0))
        else:
            qsv_score = 0
        
        metrics_text = (
            f"Best Match: {best_match}\n"
            f"Match Overlap: {best_overlap:.4f}\n"
            f"QSV Score: {qsv_score:.6f}\n"
            f"Reference Fidelity: {reference_fidelity:.4f}\n"
            f"Input Fidelity: {input_fidelity:.4f}\n"
        )
        
        # Add improvement calculation if available
        if "improvement_factor" in reconstruction_result:
            improvement = reconstruction_result["improvement_factor"]
            metrics_text += f"Improvement: {improvement:.2%}\n"
        
        # Add method information
        if "method" in reconstruction_result:
            method = reconstruction_result["method"]
            metrics_text += f"Method: {method}"
        
        ax.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        ax.set_title("Performance Metrics")
        
        plt.tight_layout()
        return fig
    
    def _plot_radar(self, ax, labels, values, colors):
        """Helper function to create a radar chart."""
        # Number of variables
        num_vars = len(labels)
        
        # Split the circle into equal parts
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Make the plot circular
        angles += angles[:1]
        values += values[:1]
        colors += colors[:1]
        
        # Draw the axes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # Draw the labels
        plt.xticks(angles[:-1], labels)
        
        # Draw the values
        ax.set_ylim(0, 1)
        ax.plot(angles, values, color='blue', linewidth=1)
        ax.fill(angles, values, color='blue', alpha=0.25)
        
        # Add grid
        ax.grid(True)
    
    def _plot_gauge(self, ax, value, title):
        """Helper function to create a gauge chart."""
        # Gauge settings
        gauge_min = 0
        gauge_max = 1
        
        # Clear axis
        ax.clear()
        
        # Create gauge
        angles = np.linspace(0, 180, 100)
        
        # Create colored gradient
        cmap = plt.cm.RdYlGn
        norm = colors.Normalize(gauge_min, gauge_max)
        
        # Draw gauge background
        for i in range(len(angles)-1):
            ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                      fc=cmap(norm(i/len(angles))), 
                                      alpha=0.2))
        
        # Draw value pointer
        angle = 180 * (1 - (value - gauge_min) / (gauge_max - gauge_min))
        angle_rad = np.deg2rad(angle)
        x = 0.5 + 0.4 * np.cos(angle_rad)
        y = 0.5 + 0.4 * np.sin(angle_rad)
        ax.plot([0.5, x], [0.5, y], 'k-', lw=2)
        ax.add_patch(plt.Circle((0.5, 0.5), 0.05, fc='k'))
        
        # Add value text
        ax.text(0.5, 0.25, f"{value:.2f}", ha='center', fontsize=16)
        
        # Add title
        ax.text(0.5, 0.9, title, ha='center', fontsize=14)
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')