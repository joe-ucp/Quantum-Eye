from qiskit_ibm_runtime import QiskitRuntimeService
import os

print("=== CHECKING WHAT CREDENTIALS ARE ACTUALLY BEING USED ===")

# Check environment variables
print("\nEnvironment variables:")
for key in ['QISKIT_IBM_TOKEN', 'QISKIT_IBM_INSTANCE', 'QISKIT_IBM_CHANNEL']:
    val = os.getenv(key)
    print(f"{key}: {val[:20] + '...' if val and len(val) > 20 else val}")

# Check saved accounts
print("\nSaved accounts:")
try:
    saved = QiskitRuntimeService.saved_accounts()
    print(f"Found {len(saved)} saved accounts")
    for name, account in saved.items():
        print(f"  {name}: instance={account.get('instance', 'N/A')[:50]}...")
except Exception as e:
    print(f"Error: {e}")

# Try to initialize service
print("\nTrying to initialize service...")
try:
    service = QiskitRuntimeService()
    account = service.active_account()
    print(f"Active account instance: {account.get('instance', 'N/A')}")
    
    backends = service.backends()
    print(f"Available backends: {[b.name for b in backends[:3]]}")
    
except Exception as e:
    print(f"Service initialization failed: {e}")