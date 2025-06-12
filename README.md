
**Revolutionary quantum state characterization using only single-basis measurements**


> *"Just as spectroscopy revolutionized molecular analysis, Quantum Eye reveals the hidden structure of quantum states through frequency-domain analysis of measurement statistics."*

 
   🔧 Quick Start
      Installation
        bash# Clone the repository
        git clone https://github.com/joe-ucp/Quantum-Eye.git
        cd Quantum-Eye

      # Install dependencies (Python 3.8+ required)

        pip install -r requirements.txt

      # Optional: Install IBM Quantum Runtime for real hardware access

        pip install qiskit_ibm_runtime

    ⚠️    ⚠️⚠️⚠️Switching Between Simulator and Real Hardware⚠️⚠️⚠️     ⚠️

    IMPORTANT: Each test can run on either simulator or real quantum hardware by changing a single boolean:
        For Bell State Test (test_bell_real.py)
        python# Line 477 - Change this single value:
        results = test_instance.run_test(use_simulator=False)  # Real hardware
        # TO:
        results = test_instance.run_test(use_simulator=True)   # Simulator
        For 100-Qubit GHZ Test (test_ghz_real.py)
        python# Line 600 - Change this single value:
        results = test.run_test(use_simulator=False)  # Real hardware  
        # TO:
        results = test.run_test(use_simulator=True)   # Simulator
        Quick Toggle Guide:

    use_simulator=True → Fast local testing (no IBM account needed)
    use_simulator=False → Real IBM quantum hardware (requires account)

        Configure IBM Quantum (For Real Hardware Only)
        python# Save your IBM Quantum credentials (one-time setup)
        from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token="YOUR_IBM_QUANTUM_TOKEN",
        set_as_default=True
    )
    Reproduce Paper Results
    1. Run Bell State Single-Basis Test (Main Result)
    bash# Default: Runs on REAL HARDWARE (change use_simulator=True for local testing)
    python tests/test_bell_real.py
    Output Location: bell_hardware_validation_[timestamp]/

    figure_1_bell_validation.png - Reproduces Figure 1 from the paper
    validation_results.json - Detailed numerical results
    Console output shows X/Y basis prediction accuracies

    2. Run 100-Qubit GHZ Holographic Test
    bash# Default: Runs on REAL HARDWARE (change use_simulator=True for local testing)
    python tests/test_ghz_real.py

    # OR use command line flag:
    python tests/test_ghz_real.py --simulator
    Output Location: ghz_simple_[timestamp]/

    holographic_results.png - Correlation reconstruction visualization
    results.json - Full reconstruction data
    summary.txt - Human-readable report

    3. Run Complete Test Suite
    bash# Run all unit tests (uses simulator by default)
    python -m unittest discover tests

    # Run specific test class
    python -m unittest tests.test_bell_sim.BellStateSimulatorValidation

    # Verbose output
    python -m unittest discover tests -v
    Expected Results
    After running the tests, you should see:

    Bell State Test: ~95% prediction accuracy for X and Y bases using only Z measurements
    GHZ Test: ~85% correlation reconstruction fidelity using golden ratio sampling
    QSV Validation: All physical states satisfy P×S×E×Q > 0

    Quick Simulator Test (No IBM Account Needed)
    bash# 1. Edit test_bell_real.py: change use_simulator=False to True
    # 2. Run the test
    python tests/test_bell_real.py

    # Check results
    repo/bell_hardware_validation
    You should see prediction accuracies >90% for both X and Y bases, confirming that single-basis measurements contain sufficient information for cross-basis prediction.

        ## 📈 Performance Benchmarks

    | System | Traditional | Quantum Eye | Improvement |
    |--------|------------|-------------|-------------|
    | 2-qubit Bell | 4096 shots × 3 bases | 256 shots × 1 basis | 48× |
    | 100-qubit GHZ | Intractable | 8 qubits measured | ✓ Possible |

        ## 🔬 Publications

[Link to paper, arxiv, key equations] Joseph Roy and Jordan Ellison of UCP Technology LLC



## 📄 Citation

```bibtex
@article{quantum-eye-2025,
  title={Quantum Eye: Complete Quantum State Recovery from Single-Basis Measurements, Quantum Eye: Holographic State Reconstruction},
  author={Roy, Joseph and Ellison, Jordan},
  journal={arXiv preprint},
  year={2025}
}
