"""Quick test to verify brittleness metrics are computed."""

import torch
import torchvision
from pathlib import Path

from core.instrumentation import ActivationCapture
from core.freedom import FreedomEngine, FreedomConfig
from demos.compute_rigidity import load_test_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=True).to(device).eval()
image_path = Path('test_images/test_00.jpg')
input_tensor = load_test_image(image_path, device)

capture = ActivationCapture(model)
with torch.no_grad():
    logits = model(input_tensor)

freedom_config = FreedomConfig()
freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
freedom_map = freedom_engine.compute_freedom_map(capture.activation_hooks, logits)

print("\n=== Brittleness Metrics Test ===")
print(f"Layers computed: {len(freedom_map['per_layer_freedom'])}")

for layer_name, layer_data in list(freedom_map['per_layer_freedom'].items())[:3]:
    print(f"\n{layer_name}:")
    print(f"  freedom_fraction: {layer_data.get('freedom_fraction', 'N/A')}")
    if 'epsilon_to_flip_min' in layer_data:
        print(f"  epsilon_to_flip_min: {layer_data['epsilon_to_flip_min']:.6f}")
        print(f"  epsilon_to_flip_median: {layer_data['epsilon_to_flip_median']:.6f}")
        print(f"  percent_dangerous_directions: {layer_data['percent_dangerous_directions']:.2f}%")
        print(f"  margin_gap: {layer_data.get('margin_gap', 'N/A')}")
        print(f"  scale_convention: {layer_data.get('scale_convention', 'N/A')}")
        print(f"  epsilon_to_flip_units: {layer_data.get('epsilon_to_flip_units', 'N/A')}")
        print(f"  delta_margin_min: {layer_data.get('delta_margin_min', 'N/A')}")
        print(f"  delta_margin_median: {layer_data.get('delta_margin_median', 'N/A')}")
        print(f"  delta_margin_max: {layer_data.get('delta_margin_max', 'N/A')}")
    else:
        print("  (brittleness metrics not computed)")

capture.remove_hooks()

