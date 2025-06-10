"""
IBM Quantum Credentials Loader for Quantum Eye

This utility helps users configure their IBM Quantum credentials for use with
the Quantum Eye framework. It provides secure credential storage and easy
connection setup.

Usage:
1. Edit the CREDENTIALS section below with your IBM Quantum account details
2. Run this script: python utils/load_creds.py
3. Your credentials will be saved and ready for use with Quantum Eye

For IBM Quantum Cloud:
- Get your API token from: https://quantum-computing.ibm.com/account
- Your instance will look like: "ibm-q/open/main" or "ibm-q-academic/university/project"

For IBM Quantum Network (Premium):
- Use your organization's specific instance format
- Channel should be "ibm_quantum"
"""

import os
import json
import getpass
from typing import Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# EDIT YOUR CREDENTIALS HERE
# =============================================================================

# IBM Quantum Credentials Configuration
# Fill in your details below, then run this script
CREDENTIALS = {
    # Your IBM Quantum API Token (required)
    # Get this from: https://quantum-computing.ibm.com/account
    "token": "YOUR_IBM_QUANTUM_TOKEN_HERE",
    
    # Your IBM Quantum Instance (required)
    # Format examples:
    #   - IBM Cloud: "ibm-q/open/main" (free tier)
    #   - IBM Cloud: "ibm-q/academic/your-university" (academic)
    #   - IBM Quantum Network: "your-hub/your-group/your-project"
    "instance": "ibm-q/open/main",
    
    # Channel type (usually don't need to change)
    # Options: "ibm_cloud" (default) or "ibm_quantum" (for premium)
    "channel": "ibm_cloud",
    
    # Default backend to use (optional - can be changed later)
    # Popular options: "ibm_brisbane", "ibm_kyoto", "ibm_osaka"
    # Leave as None to use any available backend
    "default_backend": None,
    
    # Save credentials to Qiskit account (recommended)
    "save_account": True,
    
    # Overwrite existing saved account
    "overwrite": True
}

# =============================================================================
# CREDENTIAL MANAGEMENT FUNCTIONS
# =============================================================================

def validate_credentials(creds: Dict[str, Any]) -> bool:
    """
    Validate the provided credentials.
    
    Args:
        creds: Credentials dictionary
        
    Returns:
        True if credentials appear valid, False otherwise
    """
    # Check required fields
    if not creds.get("token") or creds["token"] == "YOUR_IBM_QUANTUM_TOKEN_HERE":
        logger.error("❌ IBM Quantum token is required. Please set your token in CREDENTIALS.")
        logger.info("   Get your token from: https://quantum-computing.ibm.com/account")
        return False
    
    if not creds.get("instance"):
        logger.error("❌ IBM Quantum instance is required. Please set your instance in CREDENTIALS.")
        logger.info("   Common format: 'ibm-q/open/main' for free tier")
        return False
    
    # Check token format (basic validation)
    token = creds["token"]
    if len(token) < 32:
        logger.warning("⚠️  Token appears to be too short. Please verify your token.")
        return False
    
    # Check instance format
    instance = creds["instance"]
    if "/" not in instance:
        logger.warning("⚠️  Instance format may be incorrect. Expected format: 'hub/group/project'")
    
    logger.info("✅ Credentials appear valid")
    return True

def interactive_credential_setup() -> Dict[str, Any]:
    """
    Interactive credential setup for users who prefer not to edit the file.
    
    Returns:
        Dictionary with user-provided credentials
    """
    print("\n" + "="*60)
    print("IBM QUANTUM CREDENTIALS SETUP")
    print("="*60)
    
    print("\n1. Get your IBM Quantum credentials:")
    print("   • Visit: https://quantum-computing.ibm.com/account")
    print("   • Copy your API token")
    print("   • Note your instance (usually 'ibm-q/open/main' for free accounts)")
    
    creds = {}
    
    # Get token
    while True:
        token = getpass.getpass("\n2. Enter your IBM Quantum API token: ").strip()
        if len(token) >= 32:
            creds["token"] = token
            break
        print("   Token appears too short. Please check and try again.")
    
    # Get instance
    print("\n3. Enter your IBM Quantum instance:")
    print("   Examples:")
    print("   • ibm-q/open/main (free tier)")
    print("   • ibm-q-academic/university/project (academic)")
    print("   • your-hub/your-group/your-project (premium)")
    
    while True:
        instance = input("   Instance: ").strip()
        if instance:
            creds["instance"] = instance
            break
        print("   Instance is required. Please enter your instance.")
    
    # Get channel
    print("\n4. Select channel type:")
    print("   1. ibm_cloud (default - for most users)")
    print("   2. ibm_quantum (for premium IBM Quantum Network)")
    
    while True:
        choice = input("   Choice (1 or 2): ").strip()
        if choice == "1" or choice == "":
            creds["channel"] = "ibm_cloud"
            break
        elif choice == "2":
            creds["channel"] = "ibm_quantum"
            break
        print("   Please enter 1 or 2.")
    
    # Optional backend
    print("\n5. Default backend (optional):")
    print("   Leave empty to use any available backend")
    print("   Popular options: ibm_brisbane, ibm_kyoto, ibm_osaka")
    
    backend = input("   Default backend (or press Enter): ").strip()
    creds["default_backend"] = backend if backend else None
    
    # Save options
    creds["save_account"] = True
    creds["overwrite"] = True
    
    return creds

def setup_ibm_quantum_service(creds: Dict[str, Any]) -> bool:
    """
    Set up IBM Quantum service with the provided credentials.
    
    Args:
        creds: Credentials dictionary
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Import IBM Quantum Runtime
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        logger.info("🔄 Setting up IBM Quantum service...")
        
        # Save account if requested
        if creds.get("save_account", True):
            logger.info("💾 Saving credentials to Qiskit account...")
            
            QiskitRuntimeService.save_account(
                channel=creds["channel"],
                token=creds["token"],
                instance=creds["instance"],
                overwrite=creds.get("overwrite", True)
            )
            
            logger.info("✅ Credentials saved successfully!")
        
        # Test connection
        logger.info("🔗 Testing connection to IBM Quantum...")
        
        service = QiskitRuntimeService(
            channel=creds["channel"],
            token=creds["token"],
            instance=creds["instance"]
        )
        
        # Get available backends
        backends = service.backends()
        backend_names = [b.name for b in backends]
        
        logger.info(f"✅ Connection successful! Found {len(backends)} available backends:")
        for name in sorted(backend_names)[:10]:  # Show first 10
            logger.info(f"   • {name}")
        
        if len(backend_names) > 10:
            logger.info(f"   ... and {len(backend_names) - 10} more")
        
        # Test default backend if specified
        if creds.get("default_backend"):
            if creds["default_backend"] in backend_names:
                logger.info(f"✅ Default backend '{creds['default_backend']}' is available")
            else:
                logger.warning(f"⚠️  Default backend '{creds['default_backend']}' not found in available backends")
                logger.info("   You can still use any of the available backends listed above")
        
        return True
        
    except ImportError:
        logger.error("❌ qiskit_ibm_runtime not installed. Install with:")
        logger.error("   pip install qiskit-ibm-runtime")
        return False
    
    except Exception as e:
        logger.error(f"❌ Failed to setup IBM Quantum service: {str(e)}")
        
        # Provide helpful error messages
        error_str = str(e).lower()
        if "token" in error_str:
            logger.info("💡 Check your API token:")
            logger.info("   • Visit: https://quantum-computing.ibm.com/account")
            logger.info("   • Copy the full token (usually 32+ characters)")
        elif "instance" in error_str:
            logger.info("💡 Check your instance format:")
            logger.info("   • Free tier: 'ibm-q/open/main'")
            logger.info("   • Academic: 'ibm-q-academic/university/project'")
        elif "channel" in error_str:
            logger.info("💡 Try different channel:")
            logger.info("   • Most users: 'ibm_cloud'")
            logger.info("   • Premium users: 'ibm_quantum'")
        
        return False

def create_quantum_eye_config(creds: Dict[str, Any]) -> str:
    """
    Create a configuration dictionary for Quantum Eye.
    
    Args:
        creds: Credentials dictionary
        
    Returns:
        JSON string with Quantum Eye configuration
    """
    config = {
        "backend_type": "real",
        "backend_name": creds.get("default_backend", "ibm_brisbane"),
        "token": creds["token"],
        "instance": creds["instance"],
        "channel": creds["channel"],
        "default_shots": 4096,
        "optimization_level": 1,
        "max_reference_qubits": 10,
        "max_transform_qubits": 6
    }
    
    return json.dumps(config, indent=2)

def test_quantum_eye_integration(creds: Dict[str, Any]) -> bool:
    """
    Test integration with Quantum Eye framework.
    
    Args:
        creds: Credentials dictionary
        
    Returns:
        True if integration test passes, False otherwise
    """
    try:
        logger.info("🔄 Testing Quantum Eye integration...")
        
        # Try to import and create adapter
        from adapters.quantum_eye_adapter import QuantumEyeAdapter
        from qiskit import QuantumCircuit
        
        # Create test configuration
        config = {
            "backend_type": "real",
            "backend_name": creds.get("default_backend") or "ibm_brisbane",
            "token": creds["token"],
            "instance": creds["instance"],
            "channel": creds["channel"],
            "default_shots": 100  # Small number for testing
        }
        
        # Create adapter
        adapter = QuantumEyeAdapter(config)
        
        # Create simple test circuit
        test_circuit = QuantumCircuit(2, 2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        test_circuit.measure([0, 1], [0, 1])
        
        # Test backend info
        backend_info = adapter.get_backend_info()
        logger.info(f"✅ Quantum Eye adapter created successfully!")
        logger.info(f"   Backend: {backend_info['name']}")
        logger.info(f"   Qubits: {backend_info['num_qubits']}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Quantum Eye components: {str(e)}")
        logger.info("💡 Make sure you're running this from the Quantum Eye root directory")
        return False
    
    except Exception as e:
        logger.error(f"❌ Quantum Eye integration test failed: {str(e)}")
        return False

def main():
    """Main credential setup function."""
    print("🚀 Quantum Eye - IBM Quantum Credentials Setup")
    print("=" * 50)
    
    # Check if user wants interactive setup
    if CREDENTIALS["token"] == "YOUR_IBM_QUANTUM_TOKEN_HERE":
        print("\n📝 Credentials not configured in file.")
        choice = input("Would you like interactive setup? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            creds = interactive_credential_setup()
        else:
            print("\n💡 Please edit the CREDENTIALS section in this file and run again.")
            return False
    else:
        creds = CREDENTIALS.copy()
    
    # Validate credentials
    if not validate_credentials(creds):
        return False
    
    # Setup IBM Quantum service
    if not setup_ibm_quantum_service(creds):
        return False
    
    # Test Quantum Eye integration
    if not test_quantum_eye_integration(creds):
        logger.warning("⚠️  Quantum Eye integration test failed, but IBM Quantum setup succeeded")
        logger.info("   You can still use IBM Quantum directly")
    
    # Save example config
    config_example = create_quantum_eye_config(creds)
    config_path = os.path.join(os.path.dirname(__file__), "quantum_eye_config.json")
    
    try:
        with open(config_path, 'w') as f:
            f.write(config_example)
        logger.info(f"💾 Example Quantum Eye config saved to: {config_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not save config file: {str(e)}")
    
    # Success message
    print("\n" + "="*60)
    print("🎉 SUCCESS! Your IBM Quantum credentials are ready!")
    print("="*60)
    print("\n✅ What's working:")
    print("   • IBM Quantum Runtime connection")
    print("   • Credentials saved to Qiskit account")
    print("   • Ready for use with Quantum Eye")
    
    print("\n🚀 Next steps:")
    print("   • Run Bell state test: python tests/test_bell_real.py")
    print("   • Check available backends in IBM Quantum dashboard")
    print("   • Try other Quantum Eye examples")
    
    print(f"\n📊 Your setup:")
    print(f"   • Instance: {creds['instance']}")
    print(f"   • Channel: {creds['channel']}")
    if creds.get('default_backend'):
        print(f"   • Default backend: {creds['default_backend']}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        exit(1)