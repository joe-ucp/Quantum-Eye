"""
Quantum Eye Bell State Hardware Validation Test

This test reproduces the Bell state multi-basis prediction experiment from the paper:
"Quantum Eye: Complete Quantum State Recovery from Single-Basis Measurements"

Protocol:
1. Prepare Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
2. Measure 4096 times in Z basis only  
3. Apply Quantum Eye to create frequency signatures
4. Predict measurement distributions for X and Y bases
5. Verify predictions with actual X and Y measurements on hardware

Requires: IBM Quantum account with access to ibm_brisbane
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import logging
from typing import Dict, Tuple, Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_eye import QuantumEye
from adapters.quantum_eye_adapter import QuantumEyeAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantum_eye_bell_hardware")

class BellStateHardwareValidation:
    """
    Validates Quantum Eye's ability to predict multi-basis measurements
    from single-basis data using real quantum hardware (IBM Brisbane).
    """
    
    def __init__(self):
        """Initialize the validation test."""
        # Create output directory
        self.output_dir = f"bell_hardware_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure for IBM Brisbane
        self.backend_name = "ibm_brisbane"
        
        logger.info("=" * 80)
        logger.info("QUANTUM EYE BELL STATE VALIDATION - IBM BRISBANE")
        logger.info("Testing multi-basis prediction from single-basis measurements")
        logger.info("=" * 80)
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run the complete Bell state validation protocol on hardware.
        
        Returns:
            Dictionary with all validation results and metrics
        """
        # Initialize adapter for IBM Brisbane
        logger.info(f"\nInitializing connection to {self.backend_name}...")
        adapter = QuantumEyeAdapter({
            'backend_type': 'real',
            'backend_name': self.backend_name,
            'default_shots': 4096
            #   'backend_type': 'fake',  
            #   'backend_name': 'manila',  
            #   'default_shots': 4096
        })
        
        # Step 1: Create and register ideal Bell state
        logger.info("\nStep 1: Creating Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        bell_circuit = self._create_bell_circuit()
        
        # Register as reference
        ref_label = "bell_phi_plus"
        adapter.register_reference_circuit(bell_circuit, ref_label)
        
        # Step 2: Measure in Z basis only (4096 shots)
        logger.info("\nStep 2: Measuring 4096 times in Z basis only")
        logger.info(f"Submitting job to {self.backend_name}...")
        
        z_result = adapter.execute_circuit(
            circuit=bell_circuit,
            shots=4096,
            mitigation_enabled=True,
            reference_label=ref_label
        )
        
        z_counts = z_result['counts']
        logger.info(f"\nZ-basis measurement results:")
        for outcome, count in sorted(z_counts.items()):
            prob = count / 4096
            logger.info(f"  |{outcome}⟩: {count} ({prob:.4f})")
        
        # Step 3: Apply Quantum Eye to get frequency signature
        logger.info("\nStep 3: Creating Quantum Eye frequency signature from Z-basis data")
        
        # Get the mitigated state
        mitigation_result = z_result.get('mitigation_result', {})
        mitigated_state = mitigation_result.get('mitigated_state')
        
        if mitigated_state is None:
            raise ValueError("Failed to get mitigated state from Quantum Eye")
        
        # Extract QSV components for logging
        ucp_identity = adapter.quantum_eye.metrics_extractor.phi(mitigated_state)
        qsv = ucp_identity['quantum_signature']
        
        logger.info("\nQuantum Signature Validation (QSV) Components:")
        logger.info(f"  P (Phase Coherence):      {qsv['P']:.4f}")
        logger.info(f"  S (State Distribution):   {qsv['S']:.4f}")
        logger.info(f"  E (Entropic Measures):    {qsv['E']:.4f}")
        logger.info(f"  Q (Quantum Correlations): {qsv['Q']:.4f}")
        logger.info(f"  QSV Score (P×S×E×Q):      {qsv['P']*qsv['S']*qsv['E']*qsv['Q']:.6f}")
        
        # Step 4: Predict X and Y basis measurements
        logger.info("\nStep 4: Predicting measurement distributions for X and Y bases")
        
        x_predicted = self._predict_basis_probabilities(mitigated_state, 'X')
        y_predicted = self._predict_basis_probabilities(mitigated_state, 'Y')
        
        logger.info("\nPredicted X-basis probabilities:")
        for outcome, prob in sorted(x_predicted.items()):
            logger.info(f"  |{outcome}⟩: {prob:.4f}")
        
        logger.info("\nPredicted Y-basis probabilities:")
        for outcome, prob in sorted(y_predicted.items()):
            logger.info(f"  |{outcome}⟩: {prob:.4f}")
        
        # Step 5: Verify with actual measurements
        logger.info("\nStep 5: Verifying predictions with actual X and Y measurements on hardware")
        
        # Measure in X basis
        logger.info(f"\nMeasuring in X basis on {self.backend_name}...")
        x_circuit = self._create_bell_circuit_x_basis()
        x_result = adapter.execute_circuit(
            circuit=x_circuit,
            shots=4096,
            mitigation_enabled=False
        )
        x_counts = x_result['counts']
        x_measured = self._counts_to_probabilities(x_counts)
        
        # Measure in Y basis
        logger.info(f"\nMeasuring in Y basis on {self.backend_name}...")
        y_circuit = self._create_bell_circuit_y_basis()
        y_result = adapter.execute_circuit(
            circuit=y_circuit,
            shots=4096,
            mitigation_enabled=False
        )
        y_counts = y_result['counts']
        y_measured = self._counts_to_probabilities(y_counts)
        
        # Calculate all metrics from the paper
        metrics = self._calculate_all_metrics(
            x_predicted, x_measured, y_predicted, y_measured,
            z_counts, mitigated_state
        )
        
        # Log results summary
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Z-basis reconstruction fidelity: {metrics['z_fidelity']:.1%}")
        logger.info(f"X-basis prediction accuracy: {metrics['x_accuracy']:.1%}")
        logger.info(f"Y-basis prediction accuracy: {metrics['y_accuracy']:.1%}")
        logger.info(f"X-basis correlation (|00⟩+|11⟩): {metrics['x_correlation']:.4f}")
        logger.info(f"Y-basis anti-correlation (|01⟩+|10⟩): {metrics['y_anticorrelation']:.4f}")
        logger.info("="*60)
        
        # Create Figure 1 visualization
        self._create_figure_1(
            x_predicted, x_measured, y_predicted, y_measured,
            metrics, z_counts, qsv
        )
        
        # Save detailed results
        results = {
            'backend': self.backend_name,
            'timestamp': datetime.now().isoformat(),
            'z_basis_counts': z_counts,
            'x_predicted': {k: float(v) for k, v in x_predicted.items()},
            'x_measured': {k: float(v) for k, v in x_measured.items()},
            'y_predicted': {k: float(v) for k, v in y_predicted.items()},
            'y_measured': {k: float(v) for k, v in y_measured.items()},
            'metrics': {k: float(v) for k, v in metrics.items()},
            'qsv_components': {k: float(v) for k, v in qsv.items()},
            'job_ids': {
                'z_basis': z_result.get('job_id', 'N/A'),
                'x_basis': x_result.get('job_id', 'N/A'),
                'y_basis': y_result.get('job_id', 'N/A')
            }
        }
        
        with open(os.path.join(self.output_dir, 'validation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {self.output_dir}/")
        
        return results
    
    def _create_bell_circuit(self) -> QuantumCircuit:
        """Create Bell state |Φ+⟩ circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    def _create_bell_circuit_x_basis(self) -> QuantumCircuit:
        """Create Bell state circuit measured in X basis."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        # Apply H gates to measure in X basis
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    def _create_bell_circuit_y_basis(self) -> QuantumCircuit:
        """Create Bell state circuit measured in Y basis."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        # Apply S†H gates to measure in Y basis
        qc.sdg(0)
        qc.h(0)
        qc.sdg(1)
        qc.h(1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    def _predict_basis_probabilities(self, state: np.ndarray, basis: str) -> Dict[str, float]:
        """Predict measurement probabilities in given basis."""
        sv = Statevector(state)
        
        if basis == 'X':
            # Apply H gates to measure in X basis
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.h(1)
            sv = sv.evolve(circuit)
        elif basis == 'Y':
            # Apply S†H gates to measure in Y basis
            circuit = QuantumCircuit(2)
            circuit.sdg(0)
            circuit.h(0)
            circuit.sdg(1)
            circuit.h(1)
            sv = sv.evolve(circuit)
        
        # Get probabilities
        probs = sv.probabilities_dict()
        return probs
    
    def _counts_to_probabilities(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Convert counts to probability distribution."""
        total = sum(counts.values())
        return {outcome: count/total for outcome, count in counts.items()}
    
    def _calculate_all_metrics(self, x_predicted, x_measured, y_predicted, y_measured,
                              z_counts, mitigated_state) -> Dict[str, float]:
        """Calculate all metrics mentioned in the paper."""
        # Prediction accuracies (fidelity)
        x_accuracy = self._calculate_prediction_accuracy(x_predicted, x_measured)
        y_accuracy = self._calculate_prediction_accuracy(y_predicted, y_measured)
        
        # Quantum correlations
        x_correlation = x_predicted.get('00', 0) + x_predicted.get('11', 0)
        y_anticorrelation = y_predicted.get('01', 0) + y_predicted.get('10', 0)
        
        # Z-basis reconstruction fidelity
        z_probs = self._counts_to_probabilities(z_counts)
        ideal_bell_probs = {'00': 0.5, '01': 0.0, '10': 0.0, '11': 0.5}
        z_fidelity = self._calculate_prediction_accuracy(z_probs, ideal_bell_probs)
        
        # State purity
        purity = np.abs(np.vdot(mitigated_state, mitigated_state))
        
        # Entanglement measures
        concurrence = self._calculate_concurrence(mitigated_state)
        
        return {
            'x_accuracy': x_accuracy,
            'y_accuracy': y_accuracy,
            'x_correlation': x_correlation,
            'y_anticorrelation': y_anticorrelation,
            'z_fidelity': z_fidelity,
            'state_purity': purity,
            'concurrence': concurrence
        }
    
    def _calculate_prediction_accuracy(self, predicted: Dict[str, float], 
                                     measured: Dict[str, float]) -> float:
        """Calculate prediction accuracy using fidelity."""
        all_outcomes = set(predicted.keys()) | set(measured.keys())
        
        fidelity = 0.0
        for outcome in all_outcomes:
            p = predicted.get(outcome, 0.0)
            m = measured.get(outcome, 0.0)
            fidelity += np.sqrt(p * m)
        
        return fidelity ** 2
    
    def _calculate_concurrence(self, state: np.ndarray) -> float:
        """Calculate concurrence for two-qubit state."""
        if len(state) != 4:
            return 0.0
        
        # Reshape to 2x2 matrix
        psi = state.reshape(2, 2)
        
        # Calculate spin-flipped state
        sigma_y = np.array([[0, -1j], [1j, 0]])
        psi_tilde = np.kron(sigma_y, sigma_y) @ np.conj(state)
        
        # Calculate concurrence
        inner_product = np.abs(np.vdot(state, psi_tilde))
        concurrence = 2 * inner_product
        
        return float(concurrence)
    
    def _create_figure_1(self, x_predicted, x_measured, y_predicted, y_measured,
                        metrics, z_counts, qsv):
        """Create publication-quality Figure 1 from the paper."""
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], width_ratios=[1, 1, 1],
                             hspace=0.3, wspace=0.3)
        
        # Main comparison plots
        ax_x = fig.add_subplot(gs[0, :2])
        ax_y = fig.add_subplot(gs[1, :2])
        
        # QSV components plot
        ax_qsv = fig.add_subplot(gs[0, 2])
        
        # Metrics summary
        ax_metrics = fig.add_subplot(gs[1, 2])
        
        # Text summary at bottom
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')
        
        # Plot X-basis comparison
        self._plot_basis_comparison(ax_x, x_predicted, x_measured, 'X', metrics['x_accuracy'])
        
        # Plot Y-basis comparison
        self._plot_basis_comparison(ax_y, y_predicted, y_measured, 'Y', metrics['y_accuracy'])
        
        # Plot QSV components
        self._plot_qsv_components(ax_qsv, qsv)
        
        # Plot metrics summary
        self._plot_metrics_summary(ax_metrics, metrics)
        
        # Add summary text
        summary_text = (
            f"QUANTUM EYE BELL STATE VALIDATION ON {self.backend_name.upper()}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Protocol: Measured Bell state |Φ⁺⟩ in Z basis only (4096 shots), then predicted X and Y basis outcomes using Quantum Eye frequency signatures\n"
            f"Results: X-basis accuracy = {metrics['x_accuracy']:.1%} (paper: 97.1%), Y-basis accuracy = {metrics['y_accuracy']:.1%} (paper: 96.5%)\n"
            f"Quantum correlations preserved: X-basis (|00⟩+|11⟩) = {metrics['x_correlation']:.3f}, Y-basis (|01⟩+|10⟩) = {metrics['y_anticorrelation']:.3f}"
        )
        
        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, ha='center', va='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Overall title
        fig.suptitle('Quantum Eye: Multi-Basis Prediction from Single-Basis Measurements', 
                    fontsize=18, fontweight='bold')
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'figure_1_bell_validation.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save PDF
        plt.close()
        
        logger.info(f"\nFigure 1 saved to {fig_path}")
    
    def _plot_basis_comparison(self, ax, predicted, measured, basis, accuracy):
        """Plot comparison between predicted and measured probabilities."""
        outcomes = ['00', '01', '10', '11']
        pred_vals = [predicted.get(o, 0) for o in outcomes]
        meas_vals = [measured.get(o, 0) for o in outcomes]
        
        # Ideal Bell state values
        if basis == 'X':
            ideal_vals = [0.5, 0, 0, 0.5]  # |00⟩ and |11⟩
            title = f'X-Basis Predictions vs Measurements (Accuracy: {accuracy:.1%})'
        else:
            ideal_vals = [0, 0.5, 0.5, 0]  # |01⟩ and |10⟩
            title = f'Y-Basis Predictions vs Measurements (Accuracy: {accuracy:.1%})'
        
        x = np.arange(len(outcomes))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, meas_vals, width, label='Hardware', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x, pred_vals, width, label='Predicted', alpha=0.8, color='#ff7f0e')
        bars3 = ax.bar(x + width, ideal_vals, width, label='Ideal Bell', alpha=0.8, color='#2ca02c')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'|{o}⟩' for o in outcomes])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 0.7)
        ax.grid(True, alpha=0.3)
        
        # Add correlation info
        if basis == 'X':
            corr = predicted.get('00', 0) + predicted.get('11', 0)
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            anticorr = predicted.get('01', 0) + predicted.get('10', 0)
            ax.text(0.05, 0.95, f'Anti-correlation: {anticorr:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_qsv_components(self, ax, qsv):
        """Plot QSV components as a radar chart."""
        # Components
        categories = ['P\n(Phase)', 'S\n(State)', 'E\n(Entropy)', 'Q\n(Quantum)']
        values = [qsv['P'], qsv['S'], qsv['E'], qsv['Q']]
        
        # Create bar chart (simpler than radar for clarity)
        x = np.arange(len(categories))
        bars = ax.bar(x, values, alpha=0.8, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add QSV score
        qsv_score = qsv['P'] * qsv['S'] * qsv['E'] * qsv['Q']
        ax.text(0.5, 0.95, f'QSV Score: {qsv_score:.6f}',
               transform=ax.transAxes, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_ylabel('Component Value', fontsize=12)
        ax.set_title('Quantum Signature Components', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary(self, ax, metrics):
        """Plot key metrics as a clean summary."""
        # Prepare metrics for display
        metric_names = [
            'Z Fidelity',
            'X Accuracy',
            'Y Accuracy',
            'Concurrence',
            'State Purity'
        ]
        
        metric_values = [
            metrics['z_fidelity'],
            metrics['x_accuracy'],
            metrics['y_accuracy'],
            metrics['concurrence'],
            metrics['state_purity']
        ]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metric_names))
        bars = ax.barh(y_pos, metric_values, alpha=0.8)
        
        # Color bars based on value
        for bar, val in zip(bars, metric_values):
            if val > 0.95:
                bar.set_color('#2ecc71')  # Green for excellent
            elif val > 0.90:
                bar.set_color('#3498db')  # Blue for good
            else:
                bar.set_color('#e74c3c')  # Red for needs improvement
            
            # Add value label
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{val:.3f}', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')

import unittest

class TestBellStateHardwareValidation(unittest.TestCase):
    """Test class wrapper for unittest compatibility."""
    
    def test_bell_state_validation(self):
        """Run the Bell state validation test."""
        validator = BellStateHardwareValidation()
        results = validator.run_validation()
        
        # Add some assertions to make it a proper test
        self.assertIsNotNone(results)
        self.assertIn('x_accuracy', results['metrics'])
        self.assertIn('y_accuracy', results['metrics'])
        
        # Check that accuracies are reasonable
        self.assertGreater(results['metrics']['x_accuracy'], 0.5)
        self.assertGreater(results['metrics']['y_accuracy'], 0.5)
        
        print(f"Test completed successfully!")
        print(f"X-basis accuracy: {results['metrics']['x_accuracy']:.1%}")
        print(f"Y-basis accuracy: {results['metrics']['y_accuracy']:.1%}")

if __name__ == '__main__':
    # Run the validation test
    validator = BellStateHardwareValidation()
    results = validator.run_validation()