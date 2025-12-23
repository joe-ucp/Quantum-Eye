"""Test 1: Layer Fragility Test (Ablation-Confirmation).

Verifies that low-freedom layers break earlier than high-freedom layers
when applying the same-sized ablation.
"""

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import random

from core.instrumentation import ActivationCapture
from core.freedom import FreedomEngine, FreedomConfig, create_tail_function, select_competitor_set
from core.rigidity import is_vit_model
from demos.compute_rigidity import load_test_image


def setup_deterministic(seed: int = 42):
    """Set up deterministic behavior for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def is_valid_layer(
    model: nn.Module,
    layer_name: str,
    activation_hooks: Dict[str, Any],
    original_input: torch.Tensor,
    original_logits: torch.Tensor,
    original_top1: int,
    is_vit: bool
) -> Tuple[bool, Optional[str]]:
    """
    Check if a layer is valid for testing (systematic exclusion).
    
    Returns:
        (is_valid, error_message)
    """
    # Check if layer has activations
    if layer_name not in activation_hooks:
        return False, "Layer not in activation hooks"
    
    hook = activation_hooks[layer_name]
    if not hook.activations:
        return False, "No activations captured"
    
    # Get activation
    h = hook.activations[0]
    
    # Handle ViT blocks: extract CLS token if needed (same as FreedomEngine does)
    if is_vit:
        for name, module in model.named_modules():
            if name == layer_name:
                module_str = str(type(module)).lower()
                is_vit_block = 'encoderblock' in module_str or 'transformerblock' in module_str
                if is_vit_block and len(h.shape) == 3:
                    # Extract CLS token [B, T, F] -> [B, F]
                    h = h[:, 0, :]
                break
    
    # Check epsilon=0 baseline
    try:
        config_zero = FreedomConfig(epsilon_relative=0.0, deterministic_seed=42)
        engine_zero = FreedomEngine(model, config_zero, original_input=original_input)
        freedom_map_zero = engine_zero.compute_freedom_map(
            activation_hooks={layer_name: hook},
            logits=original_logits
        )
        
        if layer_name in freedom_map_zero['per_layer_freedom']:
            freedom_zero = freedom_map_zero['per_layer_freedom'][layer_name]['freedom_fraction']
            if abs(freedom_zero - 1.0) > 1e-6:
                return False, f"Epsilon=0 baseline failed: freedom={freedom_zero} (expected 1.0)"
        else:
            return False, "Layer not in freedom map with epsilon=0"
    except Exception as e:
        return False, f"Epsilon=0 baseline failed with exception: {str(e)}"
    
    # Check if tail function can be created
    try:
        competitor_indices = select_competitor_set(
            original_logits,
            original_top1,
            10  # Default top_m
        )
        tail_fn = create_tail_function(
            model,
            layer_name,
            original_input,
            competitor_indices,
            is_vit
        )
        # Test that tail function works
        # For ViT, h should already be CLS token [B, D] after extraction above
        h_test = h.clone().detach().requires_grad_(True)
        _ = tail_fn(h_test)
    except Exception as e:
        return False, f"Tail function creation failed: {str(e)}"
    
    # Check if freedom is exactly 0.0 for all epsilons (likely seam issue)
    try:
        configs = [
            FreedomConfig(epsilon_relative=0.01, deterministic_seed=42),
            FreedomConfig(epsilon_relative=0.05, deterministic_seed=42),
            FreedomConfig(epsilon_relative=0.1, deterministic_seed=42)
        ]
        all_zero = True
        for config in configs:
            engine = FreedomEngine(model, config, original_input=original_input)
            freedom_map = engine.compute_freedom_map(
                activation_hooks={layer_name: hook},
                logits=original_logits
            )
            if layer_name in freedom_map['per_layer_freedom']:
                freedom = freedom_map['per_layer_freedom'][layer_name]['freedom_fraction']
                if freedom != 0.0:
                    all_zero = False
                    break
        
        if all_zero:
            return False, "Freedom is exactly 0.0 for all epsilon values"
    except Exception:
        # If this check fails, don't exclude the layer
        pass
    
    return True, None


def generate_ablation_mask(
    h: torch.Tensor,
    p_percent: float,
    seed: int
) -> torch.Tensor:
    """
    Generate deterministic ablation mask.
    
    Args:
        h: Activation tensor
        p_percent: Percentage of dimensions to ablate (0.0 to 1.0)
        seed: Random seed for determinism
    
    Returns:
        Boolean mask tensor (True = keep, False = ablate)
    """
    # Flatten activation
    h_flat = h.flatten()
    num_dims = h_flat.numel()
    num_ablate = int(num_dims * p_percent)
    
    # Generate deterministic indices
    rng = torch.Generator(device=h.device)
    rng.manual_seed(seed)
    
    # Sample indices to ablate
    indices = torch.randperm(num_dims, generator=rng, device=h.device)[:num_ablate]
    
    # Create mask
    mask = torch.ones(num_dims, dtype=torch.bool, device=h.device)
    mask[indices] = False
    
    # Reshape to match h
    return mask.reshape(h.shape)


def test_layer_fragility(
    image_path: Path,
    model_name: str,
    ablation_percentages: List[float] = [0.05, 0.10, 0.20],
    num_masks_per_budget: int = 10,
    run_negative_control: bool = False
) -> Dict[str, Any]:
    """
    Test that low-freedom layers break earlier than high-freedom layers.
    
    Args:
        image_path: Path to test image
        model_name: 'resnet18' or 'vit_b_16'
        ablation_percentages: List of ablation percentages to test
        num_masks_per_budget: Number of masks to generate per ablation budget
        run_negative_control: Whether to run negative control (shuffle freedom values)
    
    Returns:
        Test results dictionary
    """
    setup_deterministic(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
    else:
        return {'status': 'FAIL', 'errors': [f'Unknown model: {model_name}']}
    
    model = model.to(device)
    model.eval()
    is_vit = is_vit_model(model)
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    # Capture activations and compute baseline freedom
    capture = ActivationCapture(model, layer_names=None)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    logits = output[0] if isinstance(output, tuple) else output
    original_top1 = int(torch.argmax(logits, dim=1)[0].item())
    
    # Compute baseline freedom map
    freedom_config = FreedomConfig(deterministic_seed=42)
    freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
    
    try:
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
    except Exception as e:
        capture.remove_hooks()
        return {'status': 'FAIL', 'errors': [f'Failed to compute freedom map: {str(e)}']}
    
    # Filter to valid layers with brittleness metrics
    valid_layers = {}
    for layer_name in freedom_map['per_layer_freedom'].keys():
        is_valid, error_msg = is_valid_layer(
            model, layer_name, capture.activation_hooks,
            input_tensor, logits, original_top1, is_vit
        )
        if is_valid:
            layer_data = freedom_map['per_layer_freedom'][layer_name]
            # Use epsilon_to_flip_min if available, fallback to freedom_fraction
            if 'epsilon_to_flip_min' in layer_data and layer_data['epsilon_to_flip_min'] != float('inf'):
                valid_layers[layer_name] = layer_data['epsilon_to_flip_min']
            elif 'freedom_fraction' in layer_data:
                # Fallback: use freedom_fraction (inverted: high freedom = high safety)
                valid_layers[layer_name] = layer_data['freedom_fraction']
        else:
            print(f"Excluding layer {layer_name}: {error_msg}")
    
    if len(valid_layers) < 2:
        capture.remove_hooks()
        return {
            'status': 'FAIL',
            'errors': [f'Not enough valid layers: {len(valid_layers)} (need at least 2)']
        }
    
    # Identify L_high (high safety = highest epsilon_to_flip_min) and L_low (low safety = lowest epsilon_to_flip_min)
    L_high = max(valid_layers.items(), key=lambda x: x[1])[0]
    L_low = min(valid_layers.items(), key=lambda x: x[1])[0]
    
    safety_high = valid_layers[L_high]
    safety_low = valid_layers[L_low]
    
    # Verify L_high and L_low are actually different layers
    if L_high == L_low:
        capture.remove_hooks()
        return {
            'status': 'SKIP',
            'errors': [f'L_high and L_low are the same layer ({L_high}) - cannot test predictive power']
        }
    
    # Verify epsilon_to_flip_min values differ meaningfully (ratio > 1.2)
    # This ensures the test has sufficient power to detect differences
    if safety_high > 0 and safety_low > 0:
        ratio = safety_high / safety_low if safety_low > 0 else float('inf')
        min_ratio = 1.2
        if ratio < min_ratio:
            capture.remove_hooks()
            return {
                'status': 'SKIP',
                'details': {
                    'L_high': L_high,
                    'L_low': L_low,
                    'safety_high': safety_high,
                    'safety_low': safety_low,
                    'ratio': ratio,
                    'min_ratio': min_ratio
                },
                'errors': [
                    f'epsilon_to_flip_min values too similar: {safety_high:.6f} vs {safety_low:.6f} '
                    f'(ratio: {ratio:.3f} < {min_ratio:.1f}) - test underpowered'
                ]
            }
    elif safety_high == float('inf') or safety_low == float('inf'):
        # If either is inf, we can't compute ratio meaningfully
        # But we can still test if one is inf and the other isn't
        if safety_high == float('inf') and safety_low != float('inf'):
            # L_high has no dangerous directions, L_low does - this is meaningful
            pass
        elif safety_low == float('inf') and safety_high != float('inf'):
            # L_low has no dangerous directions, L_high does - this is unexpected but testable
            pass
        else:
            # Both are inf - no dangerous directions in either
            capture.remove_hooks()
            return {
                'status': 'SKIP',
                'errors': ['Both L_high and L_low have no dangerous directions (epsilon_to_flip_min=inf) - cannot test predictive power']
            }
    
    # For backward compatibility, also store as freedom_high/freedom_low
    freedom_high = safety_high
    freedom_low = safety_low
    
    # Test with original safety values (using epsilon_to_flip_min)
    results = test_ablation_at_layers(
        model, capture.activation_hooks, input_tensor, logits, original_top1,
        {L_high: safety_high, L_low: safety_low},
        ablation_percentages, num_masks_per_budget, is_vit, device
    )
    
    results['L_high'] = L_high
    results['L_low'] = L_low
    results['freedom_high'] = freedom_high  # For backward compatibility
    results['freedom_low'] = freedom_low  # For backward compatibility
    results['safety_high'] = safety_high  # epsilon_to_flip_min for L_high
    results['safety_low'] = safety_low  # epsilon_to_flip_min for L_low
    results['num_valid_layers'] = len(valid_layers)
    
    # Negative control: shuffle epsilon_to_flip_min values across layers
    # This tests that the metric itself has predictive power, not just that "some layers are fragile"
    negative_control_results = None
    if run_negative_control:
        # Shuffle epsilon_to_flip_min values deterministically across layers
        layer_names = list(valid_layers.keys())
        safety_values = list(valid_layers.values())  # These are epsilon_to_flip_min values
        np.random.seed(42)
        np.random.shuffle(safety_values)
        shuffled_safety = dict(zip(layer_names, safety_values))
        
        # Find new L_high and L_low with SHUFFLED values
        # The key: we're testing if randomly assigned epsilon_to_flip_min values still predict fragility
        # If they do, then the metric has no predictive power - we're just testing layer fragility
        L_high_shuffled = max(shuffled_safety.items(), key=lambda x: x[1])[0]
        L_low_shuffled = min(shuffled_safety.items(), key=lambda x: x[1])[0]
        
        # Test with shuffled values - predictive advantage should VANISH
        negative_control_results = test_ablation_at_layers(
            model, capture.activation_hooks, input_tensor, logits, original_top1,
            {L_high_shuffled: shuffled_safety[L_high_shuffled],
             L_low_shuffled: shuffled_safety[L_low_shuffled]},
            ablation_percentages, num_masks_per_budget, is_vit, device
        )
        negative_control_results['L_high_shuffled'] = L_high_shuffled
        negative_control_results['L_low_shuffled'] = L_low_shuffled
        negative_control_results['safety_high_shuffled'] = shuffled_safety[L_high_shuffled]
        negative_control_results['safety_low_shuffled'] = shuffled_safety[L_low_shuffled]
        negative_control_results['note'] = 'Shuffled epsilon_to_flip_min values - predictive advantage should disappear'
    
    capture.remove_hooks()
    
    # Determine pass/fail
    # For most p levels: flip_rate(L_low) > flip_rate(L_high)
    passes = 0
    total = 0
    for p in ablation_percentages:
        if p in results['flip_rates']:
            flip_high = results['flip_rates'][p][L_high]
            flip_low = results['flip_rates'][p][L_low]
            total += 1
            if flip_low > flip_high:
                passes += 1
    
    status = 'PASS' if passes >= (total * 0.67) else 'FAIL'  # At least 2/3 of budgets must pass
    
    return {
        'status': status,
        'details': results,
        'negative_control': negative_control_results,
        'pass_criterion': f'{passes}/{total} ablation budgets show L_low > L_high',
        'errors': [] if status == 'PASS' else [f'Only {passes}/{total} ablation budgets show expected ordering']
    }


def test_ablation_at_layers(
    model: nn.Module,
    activation_hooks: Dict[str, Any],
    input_tensor: torch.Tensor,
    original_logits: torch.Tensor,
    original_top1: int,
    layers_to_test: Dict[str, float],
    ablation_percentages: List[float],
    num_masks_per_budget: int,
    is_vit: bool,
    device: torch.device
) -> Dict[str, Any]:
    """
    Test ablation at specific layers.
    
    Returns:
        Dictionary with flip_rates per layer per ablation percentage
    """
    flip_rates = {p: {} for p in ablation_percentages}
    
    for layer_name, freedom_value in layers_to_test.items():
        if layer_name not in activation_hooks:
            continue
        
        hook = activation_hooks[layer_name]
        if not hook.activations:
            continue
        
        h = hook.activations[0]  # Original activation
        
        # Handle ViT blocks: extract CLS token if needed
        if is_vit:
            for name, module in model.named_modules():
                if name == layer_name:
                    module_str = str(type(module)).lower()
                    is_vit_block = 'encoderblock' in module_str or 'transformerblock' in module_str
                    if is_vit_block and len(h.shape) == 3:
                        h = h[:, 0, :]  # Extract CLS token
                    break
        
        # Get competitor set and create tail function
        competitor_indices = select_competitor_set(
            original_logits,
            original_top1,
            10  # Default top_m
        )
        
        try:
            tail_fn = create_tail_function(
                model,
                layer_name,
                input_tensor,
                competitor_indices,
                is_vit
            )
        except Exception as e:
            print(f"Warning: Failed to create tail function for {layer_name}: {e}")
            continue
        
        # Test each ablation percentage
        for p in ablation_percentages:
            flips = 0
            total = 0
            
            for mask_idx in range(num_masks_per_budget):
                # Generate deterministic mask
                mask = generate_ablation_mask(h, p, seed=42 + mask_idx)
                
                # Apply ablation
                h_ablated = h.clone() * mask.float()
                h_ablated = h_ablated.requires_grad_(True)
                
                # Re-run tail function
                try:
                    with torch.no_grad():
                        competitor_logits_ablated = tail_fn(h_ablated)
                    
                    # Get full logits (approximate from competitor logits)
                    # We need to check if top1 is still top1
                    # For simplicity, check if top1 is in competitor set and has highest logit
                    top1_idx_in_competitor = competitor_indices.index(original_top1)
                    logit_top1 = competitor_logits_ablated[0, top1_idx_in_competitor].item()
                    
                    # Check if any other competitor has higher logit
                    decision_flipped = False
                    for i, class_idx in enumerate(competitor_indices):
                        if class_idx != original_top1:
                            if competitor_logits_ablated[0, i].item() >= logit_top1:
                                decision_flipped = True
                                break
                    
                    total += 1
                    if decision_flipped:
                        flips += 1
                        
                except Exception as e:
                    print(f"Warning: Ablation test failed for {layer_name} at p={p}, mask={mask_idx}: {e}")
                    continue
            
            if total > 0:
                flip_rates[p][layer_name] = flips / total
            else:
                flip_rates[p][layer_name] = 0.0
    
    return {'flip_rates': flip_rates}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 1: Layer Fragility Test')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'])
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--negative-control', action='store_true', help='Run negative control test')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        exit(1)
    
    results = test_layer_fragility(
        image_path,
        args.model,
        run_negative_control=args.negative_control
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

