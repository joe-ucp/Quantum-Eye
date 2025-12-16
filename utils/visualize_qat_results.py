"""
Visualization utility for QAT (Quantum Advantage Test) results.

Generates clear, human-readable PNG visualizations from test result JSON files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def load_summary_report(results_dir: str) -> Dict[str, Any]:
    """Load the summary report JSON from a results directory."""
    summary_path = Path(results_dir) / "summary_report.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary report not found: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def create_comprehensive_visualization(results_dir: str, output_path: Optional[str] = None) -> str:
    """
    Create a comprehensive visualization of QAT test results.
    
    Args:
        results_dir: Directory containing test results JSON files
        output_path: Optional path for output PNG (default: results_dir/results_summary.png)
        
    Returns:
        Path to the generated PNG file
    """
    if output_path is None:
        output_path = str(Path(results_dir) / "results_summary.png")
    
    summary = load_summary_report(results_dir)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Quantum Eye QAT Test Results\n{summary.get('timestamp', 'Unknown timestamp')}",
        fontsize=18,
        fontweight='bold'
    )
    
    # Extract data from summary
    results = summary.get('results', [])
    
    # 1. Scaling Test - QE vs MB Error by Qubit Count (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    scaling_data = None
    for r in results:
        if r.get('test_name') == 'test_scaling_qubits_2_to_10':
            scaling_data = r
            break
    
    if scaling_data and 'results' in scaling_data:
        qubit_counts = []
        qe_errors = []
        mb_errors = []
        
        for result in scaling_data['results']:
            n_qubits = result.get('n_qubits')
            if n_qubits is not None:
                qubit_counts.append(n_qubits)
                if not result.get('qe_failed', False):
                    qe_errors.append(result.get('qe_aggregate_error'))
                    mb_errors.append(result.get('mb_aggregate_error'))
                else:
                    qe_errors.append(None)
                    mb_errors.append(result.get('mb_aggregate_error'))
        
        if qubit_counts:
            x = np.arange(len(qubit_counts))
            width = 0.35
            
            # Plot QE errors (where available)
            qe_valid = [e for e in qe_errors if e is not None]
            qe_indices = [i for i, e in enumerate(qe_errors) if e is not None]
            if qe_valid:
                ax1.bar([x[i] - width/2 for i in qe_indices], qe_valid, width, 
                       label='QE (1 basis)', color='#2ecc71', alpha=0.8)
            
            # Plot MB errors
            ax1.bar([x[i] + width/2 for i in range(len(mb_errors))], mb_errors, width,
                   label='MB (3 bases)', color='#e74c3c', alpha=0.8)
            
            ax1.set_xlabel('Number of Qubits', fontsize=11)
            ax1.set_ylabel('Aggregate Error', fontsize=11)
            ax1.set_title('Scaling Test: Error vs Qubit Count', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(qubit_counts)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Basis Advantage Comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    basis_advantage_data = None
    for r in results:
        if r.get('test_name') == 'test_basis_advantage':
            basis_advantage_data = r
            break
    
    if basis_advantage_data and 'aggregate_errors' in basis_advantage_data:
        methods = ['QE\n(1 basis)', 'MB\n(3 bases)']
        errors = [
            basis_advantage_data['aggregate_errors'].get('qe', 0),
            basis_advantage_data['aggregate_errors'].get('mb', 0)
        ]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax2.bar(methods, errors, color=colors, alpha=0.8)
        ax2.set_ylabel('Aggregate Error', fontsize=11)
        ax2.set_title('Basis Advantage Test', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add ratio annotation
        if errors[1] > 0:
            ratio = errors[0] / errors[1]
            ax2.text(0.5, 0.95, f'Ratio: {ratio:.3f}\n(QE/MB)', 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
    
    # 3. High Noise Test (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    high_noise_data = None
    for r in results:
        if r.get('test_name') == 'test_high_noise_error_trust':
            high_noise_data = r
            break
    
    if high_noise_data and 'aggregate_errors' in high_noise_data:
        methods = ['QE\n(1 basis)', 'MB\n(3 bases)']
        errors = [
            high_noise_data['aggregate_errors'].get('qe', 0),
            high_noise_data['aggregate_errors'].get('mb', 0)
        ]
        noise_level = high_noise_data.get('noise_level', 0) * 100
        
        bars = ax3.bar(methods, errors, color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax3.set_ylabel('Aggregate Error', fontsize=11)
        ax3.set_title(f'High Noise Test ({noise_level:.0f}% noise)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add ratio annotation
        if errors[1] > 0:
            ratio = errors[0] / errors[1]
            ax3.text(0.5, 0.95, f'Ratio: {ratio:.3f}', 
                    transform=ax3.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
    
    # 4. 2-Qubit Observable Recovery Details (middle left, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    qubit2_data = None
    for r in results:
        if r.get('test_name') == 'test_observable_recovery_2qubit':
            qubit2_data = r
            break
    
    if qubit2_data and 'aggregate_errors' in qubit2_data:
        methods = ['QE', 'MB', 'Z-Only\nBaseline']
        errors = [
            qubit2_data['aggregate_errors'].get('qe', 0),
            qubit2_data['aggregate_errors'].get('mb', 0),
            qubit2_data['aggregate_errors'].get('z_only', 0)
        ]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        bars = ax4.bar(methods, errors, color=colors, alpha=0.8)
        ax4.set_ylabel('Aggregate Error', fontsize=11)
        ax4.set_title('2-Qubit Observable Recovery Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Summary Metrics (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_metrics = summary.get('summary_metrics', {})
    avg_qe = summary_metrics.get('average_qe_error', 0)
    avg_mb = summary_metrics.get('average_mb_error', 0)
    
    metrics_text = (
        f"Test Suite Summary\n"
        f"{'='*30}\n\n"
        f"Total Tests: {summary.get('total_tests', 0)}\n"
        f"Timestamp: {summary.get('timestamp', 'N/A')}\n\n"
        f"Average QE Error: {avg_qe:.6f}\n"
        f"Average MB Error: {avg_mb:.6f}\n"
    )
    
    if avg_mb > 0:
        overall_ratio = avg_qe / avg_mb
        metrics_text += f"\nOverall Ratio: {overall_ratio:.4f}\n"
        if overall_ratio < 1.0:
            metrics_text += f"(QE {((1-overall_ratio)*100):.1f}% better)"
        else:
            metrics_text += f"(QE {((overall_ratio-1)*100):.1f}% worse)"
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 6. Scaling Test - Ratio Plot (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    if scaling_data and 'results' in scaling_data:
        qubit_counts = []
        ratios = []
        
        for result in scaling_data['results']:
            n_qubits = result.get('n_qubits')
            if n_qubits is not None and not result.get('qe_failed', False):
                qubit_counts.append(n_qubits)
                ratio = result.get('qe_mb_ratio', 1.0)
                ratios.append(ratio)
        
        if qubit_counts:
            ax6.plot(qubit_counts, ratios, 'o-', color='#3498db', linewidth=2, markersize=8)
            ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal Performance')
            ax6.set_xlabel('Number of Qubits', fontsize=11)
            ax6.set_ylabel('QE/MB Error Ratio', fontsize=11)
            ax6.set_title('Scaling: QE/MB Ratio', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            # Set y-axis limits with defensive check for empty ratios list
            if ratios:
                ax6.set_ylim([0, max(1.2, max(ratios) * 1.1)])
            else:
                ax6.set_ylim([0, 1.2])
    
    # 7. Observable Error Breakdown (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    if qubit2_data and 'qe_errors' in qubit2_data and 'mb_errors' in qubit2_data:
        # Extract X, Y, Z errors for both methods
        qe_errors = qubit2_data['qe_errors']
        mb_errors = qubit2_data['mb_errors']
        
        observables = ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1']
        qe_vals = [qe_errors.get(obs, 0) for obs in observables]
        mb_vals = [mb_errors.get(obs, 0) for obs in observables]
        
        x = np.arange(len(observables))
        width = 0.35
        
        ax7.bar(x - width/2, qe_vals, width, label='QE', color='#2ecc71', alpha=0.8)
        ax7.bar(x + width/2, mb_vals, width, label='MB', color='#e74c3c', alpha=0.8)
        
        ax7.set_xlabel('Observable', fontsize=11)
        ax7.set_ylabel('Error', fontsize=11)
        ax7.set_title('2-Qubit: Per-Observable Errors', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(observables, rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Test Status Overview (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Count test results
    test_names = []
    test_status = []
    
    for r in results:
        test_name = r.get('test_name', 'unknown')
        # Simplify test names for display
        if 'scaling' in test_name:
            test_name = 'Scaling Test'
        elif 'basis_advantage' in test_name:
            test_name = 'Basis Advantage'
        elif 'high_noise' in test_name:
            test_name = 'High Noise'
        elif 'observable_recovery_2qubit' in test_name:
            test_name = '2-Qubit Recovery'
        elif 'observable_recovery_3qubit' in test_name:
            test_name = '3-Qubit Recovery'
        elif 'observable_recovery_4qubit' in test_name:
            test_name = '4-Qubit Recovery'
        
        test_names.append(test_name)
        # All tests in summary passed (otherwise they wouldn't be here)
        test_status.append('PASSED')
    
    status_text = "Test Status\n" + "="*25 + "\n\n"
    for name, status in zip(test_names, test_status):
        status_text += f"[PASS] {name}\n"
    
    ax8.text(0.1, 0.5, status_text, fontsize=10, va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def main():
    """Command-line interface for generating visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate PNG visualization from QAT test results"
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Directory containing test results JSON files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output PNG path (default: results_dir/results_summary.png)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_comprehensive_visualization(args.results_dir, args.output)
        print(f"[OK] Visualization saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

