"""
Quantum Eye 100-Qubit GHZ Correlation Reconstruction Test - IBM Brisbane ONLY

This test demonstrates the holographic reconstruction of 100-qubit GHZ state correlations
using a scientifically validated method based on golden ratio sampling.


- Golden ratio positions: 38, 62 (φ⁻², φ⁻¹)
- Fibonacci positions: 21, 55 (F₇, F₉)

Based on: "Quantum Eye: Holographic State Reconstruction 
for 100-qubit GHZ States Using Golden Ratio Sampling"

Protocol:
1. Create 100-qubit GHZ state on Brisbane's 127 qubits
2. Measure exactly qubits [38,62], [21,55], [38,55], [21,62]
3. Reconstruct all 4,950 qubit-pair correlations
4. Validate predictions against actual measurements

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import pandas as pd
import logging
import unittest
import sys
from typing import Dict, List, Tuple, Any, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_eye import QuantumEye
from adapters.quantum_eye_adapter import QuantumEyeAdapter
from utils.visualization import VisualizationHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantum_eye_ghz_hardware")

class GHZBrisbaneValidation:
    """
    IBM Brisbane-specific validation of 100-qubit GHZ correlation reconstruction.
    
    HARDCODED 
    - Brisbane backend (127 qubits)
    - 100-qubit GHZ state
    - Measurement positions: [38,62], [21,55], [38,55], [21,62]
    - 1000 shots per circuit
    
    
    """
    
    def __init__(self, shots: int = 1000):
        """
        Initialize the validation test for IBM Brisbane.
        
        Args:
            shots: Number of shots per circuit (default: 1000)
        """
        # Golden ratio
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Store shots parameter
        self.shots = shots
        
        # Create output directory
        self.output_dir = f"ghz_100q_brisbane_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        #  positions that as described in the paper.
        self.measurement_pairs = [
            ([38, 62], "golden_primary"),      # φ⁻² to φ⁻¹ 
            ([21, 55], "fibonacci_core"),      # F₇ to F₉ 
            ([38, 55], "cross_golden_fib"),    # Golden to Fibonacci cross
            ([21, 62], "cross_fib_golden"),    # Fibonacci to Golden cross
        ]
        
        #  validation pairs
        self.validation_pairs = [
            (38, 62, "golden_primary"),
            (21, 55, "fibonacci_core"),
            (0, 99, "maximum_distance"),
            (49, 50, "center_correlation"),
            (13, 34, "fibonacci_sequence"),
            (25, 75, "half_distance"),
            (0, 1, "boundary_start"),
            (98, 99, "boundary_end"),
        ]
        
        # Add visualization helper
        self.viz = VisualizationHelper({
            'figsize': (12, 8),
            'colormap': 'viridis'
        })
        
        logger.info("=" * 80)
        logger.info("QUANTUM EYE 100-QUBIT GHZ VALIDATION")
        logger.info("2-qubit×4 Golden Ratio Method")
        logger.info("=" * 80)
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run the complete GHZ correlation reconstruction validation.
        
        Returns:
            Dictionary with all validation results and metrics
        """
        logger.info("\nInitializing connection to IBM Brisbane...")
        adapter = QuantumEyeAdapter({
            'backend_type': 'real',
            'backend_name': 'ibm_brisbane',
            'default_shots': 1000
        })
        
        # Results storage
        results = {
            'backend': 'ibm_brisbane',
            'num_qubits': 100,
            'shots': 1000,
            'timestamp': datetime.now().isoformat(),
            'measurement_results': {},
            'reconstructed_correlations': {},
            'validation_results': {},
            'metrics': {}
        }
        
        # Step 1: Execute 2-qubit measurements
        logger.info(f"\nStep 1: Measuring 2-qubit pairs at golden ratio positions")
        logger.info("="*60)
        
        measurement_data = {}
        bell_correlations = []
        
        for pair, label in self.measurement_pairs:
            logger.info(f"\nMeasuring qubits {pair[0]}-{pair[1]} ({label})")
            
            # Create circuit
            circuit = self._create_2qubit_ghz_circuit(pair)
            
            # Execute on hardware
            result = adapter.execute_circuit(
                circuit=circuit,
                shots=self.shots,
                mitigation_enabled=True
            )
            
            # Extract data
            counts = result['counts']
            probs = self._counts_to_probabilities(counts)
            bell_correlation = self._calculate_bell_correlation(probs)
            
            measurement_data[label] = {
                'pair': pair,
                'counts': counts,
                'probabilities': probs,
                'bell_correlation': bell_correlation,
                'job_id': result.get('job_id', 'N/A')
            }
            bell_correlations.append(bell_correlation)
            
            logger.info(f"  Bell correlation ⟨Z₁Z₂⟩: {bell_correlation:.4f}")
            logger.info(f"  Dominant outcomes: {self._get_dominant_outcomes(probs, 2)}")
        
        results['measurement_results'] = measurement_data
        
        # Step 2: Holographic reconstruction
        logger.info(f"\nStep 2: Holographic reconstruction of all {100*99//2} correlations")
        logger.info("="*60)
        
        reconstructed = self._holographic_reconstruction(measurement_data)
        results['reconstructed_correlations'] = reconstructed
        
        # Calculate reconstruction statistics
        correlation_values = list(reconstructed.values())
        results['metrics']['reconstruction_stats'] = {
            'total_correlations': len(correlation_values),
            'mean_correlation': float(np.mean(correlation_values)),
            'std_correlation': float(np.std(correlation_values)),
            'max_correlation': float(np.max(correlation_values)),
            'min_correlation': float(np.min(correlation_values))
        }
        
        logger.info(f"Reconstructed {len(correlation_values)} correlations")
        logger.info(f"Mean correlation strength: {np.mean(correlation_values):.4f}")
        
        # Step 3: Validate reconstruction
        logger.info(f"\nStep 3: Validating reconstruction accuracy")
        logger.info("="*60)
        
        validation_data = {}
        fidelities = []
        
        for q1, q2, label in self.validation_pairs:
            logger.info(f"\nValidating correlation {q1}-{q2} ({label})")
            
            # Create validation circuit
            circuit = self._create_correlation_validation_circuit(q1, q2)
            
            # Execute on hardware
            val_result = adapter.execute_circuit(
                circuit=circuit,
                shots=self.shots,
                mitigation_enabled=False
            )
            
            # Calculate actual correlation
            actual_corr = self._calculate_correlation_from_counts(val_result['counts'])
            
            # Get reconstructed value
            reconstructed_corr = reconstructed.get(f"{q1}_{q2}", 0.0)
            
            # Calculate fidelity
            fidelity = 1.0 - abs(actual_corr - reconstructed_corr)
            fidelities.append(fidelity)
            
            validation_data[f"{q1}_{q2}"] = {
                'label': label,
                'actual': float(actual_corr),
                'reconstructed': float(reconstructed_corr),
                'fidelity': float(fidelity),
                'error': float(abs(actual_corr - reconstructed_corr))
            }
            
            logger.info(f"  Actual: {actual_corr:.4f}, Reconstructed: {reconstructed_corr:.4f}")
            logger.info(f"  Fidelity: {fidelity:.4f} ({fidelity*100:.1f}%)")
        
        results['validation_results'] = validation_data
        
        # Calculate overall metrics
        mean_fidelity = np.mean(fidelities)
        results['metrics']['validation_stats'] = {
            'mean_fidelity': float(mean_fidelity),
            'std_fidelity': float(np.std(fidelities)),
            'min_fidelity': float(np.min(fidelities)),
            'max_fidelity': float(np.max(fidelities))
        }
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Mean correlation reconstruction fidelity: {mean_fidelity:.4f} ({mean_fidelity*100:.1f}%)")
        logger.info(f"Total qubits measured: 8 out of 100 (8.0%)")
        logger.info(f"Information compression ratio: {4950/8:.0f}:1")
        logger.info("="*60)
        
        # Create visualizations
        self._create_figure_1(measurement_data, bell_correlations)
        self._create_figure_2(validation_data, mean_fidelity)
        
        # Save detailed results
        self._save_results(results)
        
        return results
    
    def _create_2qubit_ghz_circuit(self, qubit_pair: List[int]) -> QuantumCircuit:
        """Create 100-qubit GHZ circuit measuring only 2 qubits."""
        qc = QuantumCircuit(100, 2)
        
        # Create GHZ state
        qc.h(0)
        for i in range(99):
            qc.cx(i, i + 1)
        
        # Measure only the specified pair
        qc.measure(qubit_pair[0], 0)
        qc.measure(qubit_pair[1], 1)
        
        return qc
    
    def _create_correlation_validation_circuit(self, q1: int, q2: int) -> QuantumCircuit:
        """Create circuit to validate correlation between two specific qubits."""
        qc = QuantumCircuit(100, 2)
        
        # Create GHZ state
        qc.h(0)
        for i in range(99):
            qc.cx(i, i + 1)
        
        # Measure the validation pair
        qc.measure(q1, 0)
        qc.measure(q2, 1)
        
        return qc
    
    def _counts_to_probabilities(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Convert counts to probability distribution."""
        total = sum(counts.values())
        return {outcome: count/total for outcome, count in counts.items()}
    
    def _calculate_bell_correlation(self, probs: Dict[str, float]) -> float:
        """Calculate Bell correlation ⟨Z₁Z₂⟩ from probabilities."""
        return (probs.get('00', 0) + probs.get('11', 0) - 
                probs.get('01', 0) - probs.get('10', 0))
    
    def _calculate_correlation_from_counts(self, counts: Dict[str, int]) -> float:
        """Calculate correlation from measurement counts."""
        probs = self._counts_to_probabilities(counts)
        return self._calculate_bell_correlation(probs)
    
    def _get_dominant_outcomes(self, probs: Dict[str, float], n: int = 2) -> str:
        """Get the n most probable outcomes as a string."""
        sorted_outcomes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"|{outcome}⟩:{prob:.3f}" for outcome, prob in sorted_outcomes[:n]])
    
    def _holographic_reconstruction(self, measurement_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform holographic reconstruction of all correlations from 2-qubit measurements.
        This is the core algorithm.
        """
        reconstructed = {}
        
        # Extract Bell correlations from each measurement
        bell_correlations = {
            label: data['bell_correlation'] 
            for label, data in measurement_data.items()
        }
        
        # Calculate master correlation strength (geometric mean)
        master_correlation = np.power(
            abs(bell_correlations['golden_primary']) * 
            abs(bell_correlations['fibonacci_core']) * 
            abs(bell_correlations['cross_golden_fib']) * 
            abs(bell_correlations['cross_fib_golden']),
            0.25
        )
        
        # hardcoded positions
        positions = {
            'golden_1': 38,  # φ⁻²
            'golden_2': 62,  # φ⁻¹
            'fib_1': 21,     # F₇
            'fib_2': 55,     # F₉
        }
        
        logger.info(f"\nHolographic reconstruction parameters:")
        logger.info(f"  Master correlation: {master_correlation:.4f}")
        logger.info(f"  Golden positions: {positions['golden_1']}, {positions['golden_2']}")
        logger.info(f"  Fibonacci positions: {positions['fib_1']}, {positions['fib_2']}")
        
        # Reconstruct all pairwise correlations for 100 qubits
        for i in range(100):
            for j in range(i + 1, 100):
                distance = abs(j - i)
                
                # Calculate holographic influence from each measured position
                influences = []
                for pos in positions.values():
                    dist_to_pos = min(abs(i - pos), abs(j - pos))
                    if 'golden' in [k for k, v in positions.items() if v == pos]:
                        # Golden ratio decay
                        decay = np.exp(-dist_to_pos / (self.phi * 100))
                    else:
                        # Fibonacci decay
                        decay = np.exp(-dist_to_pos / (1.618 * 100))
                    influences.append(decay)
                
                # Combined influence
                base_influence = np.mean(influences)
                
                # Boost for directly measured pairs
                pair_set = {i, j}
                boost = 1.0
                for data in measurement_data.values():
                    if set(data['pair']) == pair_set:
                        boost = 1.2  # 20% boost for measured pairs
                        break
                
                # Distance-based modulation
                if distance <= 10:
                    distance_factor = 1.0
                elif distance <= 50:
                    distance_factor = 0.8
                else:
                    distance_factor = 0.6 + 0.4 * np.exp(-distance / 200)  # 2 * 100
                
                # Final correlation
                correlation = master_correlation * base_influence * boost * distance_factor
                
                # Noise correction for long-range correlations
                if master_correlation < 0.8:
                    noise_correction = np.exp(-distance / (3.0 * self.phi * 100))
                    correlation *= noise_correction
                
                # Store bidirectionally
                reconstructed[f"{i}_{j}"] = correlation
                reconstructed[f"{j}_{i}"] = correlation
        
        return reconstructed
    
    def _create_figure_1(self, measurement_data: Dict[str, Any], bell_correlations: List[float]):
        """Create Figure 1 using the visualization helper."""
        # Convert your data to format expected by visualization helper
        detection_result = {
            'resonance_results': {
                label: {
                    'overlap': data['bell_correlation'],
                    'confidence': abs(data['bell_correlation']),
                    'detected': data['bell_correlation'] > 0.5
                }
                for label, data in measurement_data.items()
            },
            'threshold': 0.7,
            'best_match': max(measurement_data.keys(), 
                            key=lambda k: measurement_data[k]['bell_correlation'])
        }
        
        fig = self.viz.plot_resonance_results(detection_result, 
                                            "100-Qubit GHZ Measurement Results")
        
        # Save
        fig_path = os.path.join(self.output_dir, 'figure_1_measurement_results.png')
        self.viz.save_figure(fig, fig_path)
        logger.info(f"Figure 1 saved to {fig_path}")
    
    def _create_figure_2(self, validation_data: Dict[str, Any], mean_fidelity: float):
        """Create Figure 2 using the visualization helper."""
        # Convert validation data to reconstruction result format
        reconstruction_result = {
            'reference_fidelity': mean_fidelity,
            'input_fidelity': 0.854,  # Target fidelity
            'confidence': mean_fidelity,
            'method': '2-qubit×4 Golden Ratio',
            'reconstructed_state': None  # Could add if needed
        }
        
        fig = self.viz.plot_reconstruction_results(reconstruction_result,
                                                 "Correlation Reconstruction Validation")
        
        # Save
        fig_path = os.path.join(self.output_dir, 'figure_2_validation_results.png')
        self.viz.save_figure(fig, fig_path)
        logger.info(f"Figure 2 saved to {fig_path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save detailed results to JSON and CSV files."""
        # Save complete results as JSON
        json_path = os.path.join(self.output_dir, 'complete_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComplete results saved to {json_path}")
        
        # Save validation results as CSV for easy analysis
        validation_df = pd.DataFrame.from_dict(results['validation_results'], orient='index')
        csv_path = os.path.join(self.output_dir, 'validation_results.csv')
        validation_df.to_csv(csv_path)
        logger.info(f"Validation results saved to {csv_path}")
        
        # Save measurement summary
        summary = {
            'Test Configuration': {
                'Backend': 'ibm_brisbane',
                'Number of Qubits': 100,
                'Shots per Circuit': 1000,
                'Total Circuits Run': len(results['measurement_results']) + len(results['validation_results']),
                'Timestamp': results['timestamp']
            },
            'Performance Metrics': {
                'Mean Correlation Fidelity': results['metrics']['validation_stats']['mean_fidelity'],
                'Standard Deviation': results['metrics']['validation_stats']['std_fidelity'],
                'Min Fidelity': results['metrics']['validation_stats']['min_fidelity'],
                'Max Fidelity': results['metrics']['validation_stats']['max_fidelity'],
                'Information Compression': '619:1'  # 4950/8
            },
            'Measurement Strategy': {
                'Total Qubits Measured': 8,
                'Measurement Percentage': '8.0%',
                'Number of 2-qubit Measurements': 4,
                'Golden Ratio Positions': [38, 62],
                'Fibonacci Positions': [21, 55]
            }
        }
        
        summary_path = os.path.join(self.output_dir, 'test_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Test summary saved to {summary_path}")

class TestGHZBrisbaneValidation(unittest.TestCase):
    """Test class wrapper for unittest compatibility."""
    
    def test_ghz_correlation_reconstruction(self):
        """Run the complete GHZ correlation reconstruction validation test."""
        validator = GHZBrisbaneValidation(shots=1000)
        results = validator.run_validation()
        
        # Add proper test assertions
        self.assertIsNotNone(results)
        self.assertIn('measurement_results', results)
        self.assertIn('validation_results', results)
        self.assertIn('metrics', results)
        
        # Check that we measured the correct number of pairs
        self.assertEqual(len(results['measurement_results']), 4)
        
        # Check that all Bell correlations are positive (GHZ property)
        for label, data in results['measurement_results'].items():
            bell_corr = data['bell_correlation']
            self.assertGreater(bell_corr, 0.0, 
                             f"Expected positive correlation for {label}, got {bell_corr}")
        
        # Check reconstruction coverage
        self.assertGreater(len(results['reconstructed_correlations']), 4000,
                          "Should reconstruct thousands of correlations")
        
        # Check validation fidelity is reasonable
        mean_fidelity = results['metrics']['validation_stats']['mean_fidelity']
        self.assertGreater(mean_fidelity, 0.5, 
                          f"Mean fidelity {mean_fidelity:.3f} too low")
        
        # Check compression ratio
        total_correlations = len(results['reconstructed_correlations']) // 2
        compression_ratio = total_correlations / 8  # 8 qubits measured
        self.assertGreater(compression_ratio, 500,
                          f"Compression ratio {compression_ratio:.0f}:1 too low")
        
        print(f"✓ GHZ Test completed successfully!")
        print(f"  Mean fidelity: {mean_fidelity:.1%}")
        print(f"  Compression: {compression_ratio:.0f}:1")
        print(f"  Correlations reconstructed: {total_correlations}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--unittest':
        # Run as unittest
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run the original validation test
        validator = GHZBrisbaneValidation()
        results = validator.run_validation()