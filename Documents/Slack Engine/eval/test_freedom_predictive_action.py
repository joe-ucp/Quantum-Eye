"""Test 5: Predictive Test on Real Downstream Action (Layer Selection).

Validates that freedom scores predict which layers can be safely pruned.
"""

import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import random

from core.instrumentation import ActivationCapture
from core.freedom import FreedomEngine, FreedomConfig, create_tail_function, select_competitor_set
from core.rigidity import is_vit_model
from demos.compute_rigidity import load_test_image
from test_freedom_ablation import (
    setup_deterministic,
    is_valid_layer,
    generate_ablation_mask
)


def test_predictive_action(
    image_path: Path,
    model_name: str,
    prune_percentage: float = 0.20,
    num_masks_per_strategy: int = 10,
    run_negative_control: bool = False
) -> Dict[str, Any]:
    """
    Test that freedom scores predict which layers can be safely pruned.
    
    Args:
        image_path: Path to test image
        model_name: 'resnet18' or 'vit_b_16'
        prune_percentage: Percentage of dimensions to prune (0.0 to 1.0)
        num_masks_per_strategy: Number of masks to generate per pruning strategy
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
    
    if len(valid_layers) < 3:
        capture.remove_hooks()
        return {
            'status': 'FAIL',
            'errors': [f'Not enough valid layers: {len(valid_layers)} (need at least 3)']
        }
    
    # Pick high-safety (highest epsilon_to_flip_min) and low-safety (lowest epsilon_to_flip_min) layers
    sorted_layers = sorted(valid_layers.items(), key=lambda x: x[1], reverse=True)
    
    # Top 1-2 high-safety layers (highest epsilon_to_flip_min)
    high_freedom_layers = [layer_name for layer_name, _ in sorted_layers[:2]]
    
    # Bottom 1-2 low-safety layers (lowest epsilon_to_flip_min)
    low_freedom_layers = [layer_name for layer_name, _ in sorted_layers[-2:]]
    
    # Verify layers are meaningfully different
    if len(sorted_layers) >= 2:
        top_safety = sorted_layers[0][1]
        bottom_safety = sorted_layers[-1][1]
        
        # Check ratio if both are finite
        if top_safety != float('inf') and bottom_safety != float('inf') and bottom_safety > 0:
            ratio = top_safety / bottom_safety
            min_ratio = 1.2
            if ratio < min_ratio:
                capture.remove_hooks()
                return {
                    'status': 'SKIP',
                    'details': {
                        'top_safety': top_safety,
                        'bottom_safety': bottom_safety,
                        'ratio': ratio,
                        'min_ratio': min_ratio
                    },
                    'errors': [
                        f'epsilon_to_flip_min values too similar: {top_safety:.6f} vs {bottom_safety:.6f} '
                        f'(ratio: {ratio:.3f} < {min_ratio:.1f}) - test underpowered'
                    ]
                }
    
    # Random layers (control)
    all_layer_names = list(valid_layers.keys())
    np.random.seed(42)
    random_layer_names = np.random.choice(
        all_layer_names,
        size=min(2, len(all_layer_names)),
        replace=False
    ).tolist()
    
    # Test pruning at each layer set
    results = {}
    
    for strategy_name, layer_set in [
        ('prune_high_freedom_layers', high_freedom_layers),
        ('prune_low_freedom_layers', low_freedom_layers),
        ('prune_random_layers', random_layer_names)
    ]:
        flip_rate = test_pruning_at_layers(
            model, capture.activation_hooks, input_tensor, logits, original_top1,
            layer_set, prune_percentage, num_masks_per_strategy, is_vit, device
        )
        results[strategy_name] = {
            'layers': layer_set,
            'flip_rate': flip_rate
        }
    
    # Negative control: shuffle freedom values
    negative_control_results = None
    if run_negative_control:
        # Shuffle freedom values deterministically
        layer_names = list(valid_layers.keys())
        freedom_values = list(valid_layers.values())
        np.random.seed(42)
        np.random.shuffle(freedom_values)
        shuffled_freedom = dict(zip(layer_names, freedom_values))
        
        # Pick new high/low layers with shuffled values
        sorted_shuffled = sorted(shuffled_freedom.items(), key=lambda x: x[1], reverse=True)
        high_shuffled = [layer_name for layer_name, _ in sorted_shuffled[:2]]
        low_shuffled = [layer_name for layer_name, _ in sorted_shuffled[-2:]]
        
        negative_control_results = {}
        for strategy_name, layer_set in [
            ('prune_high_freedom_layers_shuffled', high_shuffled),
            ('prune_low_freedom_layers_shuffled', low_shuffled),
            ('prune_random_layers', random_layer_names)
        ]:
            flip_rate = test_pruning_at_layers(
                model, capture.activation_hooks, input_tensor, logits, original_top1,
                layer_set, prune_percentage, num_masks_per_strategy, is_vit, device
            )
            negative_control_results[strategy_name] = {
                'layers': layer_set,
                'flip_rate': flip_rate
            }
    
    capture.remove_hooks()
    
    # Determine pass/fail
    # Expected ordering: prune_high_freedom < prune_random < prune_low_freedom
    flip_high = results['prune_high_freedom_layers']['flip_rate']
    flip_low = results['prune_low_freedom_layers']['flip_rate']
    flip_random = results['prune_random_layers']['flip_rate']
    
    errors = []
    status = 'PASS'
    
    if flip_high >= flip_random:
        errors.append(f'High-freedom pruning flip rate {flip_high:.3f} >= random {flip_random:.3f}')
        status = 'FAIL'
    
    if flip_low <= flip_random:
        errors.append(f'Low-freedom pruning flip rate {flip_low:.3f} <= random {flip_random:.3f}')
        status = 'FAIL'
    
    if flip_high >= flip_low:
        errors.append(f'High-freedom pruning flip rate {flip_high:.3f} >= low-freedom {flip_low:.3f}')
        status = 'FAIL'
    
    return {
        'status': status,
        'details': {
            'results': results,
            'expected_ordering': 'prune_high_freedom < prune_random < prune_low_freedom',
            'actual_ordering': {
                'prune_high_freedom': flip_high,
                'prune_random': flip_random,
                'prune_low_freedom': flip_low
            }
        },
        'negative_control': negative_control_results,
        'errors': errors
    }


def test_pruning_at_layers(
    model: nn.Module,
    activation_hooks: Dict[str, Any],
    input_tensor: torch.Tensor,
    original_logits: torch.Tensor,
    original_top1: int,
    layer_set: List[str],
    prune_percentage: float,
    num_masks: int,
    is_vit: bool,
    device: torch.device
) -> float:
    """
    Test pruning at a set of layers and return flip rate.
    
    Returns:
        Decision flip rate (0.0 to 1.0)
    """
    flips = 0
    total = 0
    
    for layer_name in layer_set:
        if layer_name not in activation_hooks:
            continue
        
        hook = activation_hooks[layer_name]
        if not hook.activations:
            continue
        
        h = hook.activations[0]
        
        # Handle ViT blocks: extract CLS token if needed
        if is_vit:
            for name, module in model.named_modules():
                if name == layer_name:
                    module_str = str(type(module)).lower()
                    is_vit_block = 'encoderblock' in module_str or 'transformerblock' in module_str
                    if is_vit_block and len(h.shape) == 3:
                        h = h[:, 0, :]
                    break
        
        # Get competitor set and create tail function
        competitor_indices = select_competitor_set(
            original_logits,
            original_top1,
            10
        )
        
        try:
            tail_fn = create_tail_function(
                model,
                layer_name,
                input_tensor,
                competitor_indices,
                is_vit
            )
        except Exception:
            continue
        
        # Test multiple masks
        for mask_idx in range(num_masks):
            # Generate deterministic mask
            mask = generate_ablation_mask(h, prune_percentage, seed=42 + mask_idx)
            
            # Apply pruning
            h_pruned = h.clone() * mask.float()
            h_pruned = h_pruned.requires_grad_(True)
            
            # Re-run tail function
            try:
                with torch.no_grad():
                    competitor_logits_pruned = tail_fn(h_pruned)
                
                # Check if decision flipped
                top1_idx_in_competitor = competitor_indices.index(original_top1)
                logit_top1 = competitor_logits_pruned[0, top1_idx_in_competitor].item()
                
                decision_flipped = False
                for i, class_idx in enumerate(competitor_indices):
                    if class_idx != original_top1:
                        if competitor_logits_pruned[0, i].item() >= logit_top1:
                            decision_flipped = True
                            break
                
                total += 1
                if decision_flipped:
                    flips += 1
                    
            except Exception:
                continue
    
    return flips / total if total > 0 else 0.0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 5: Predictive Action Test')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'])
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--negative-control', action='store_true', help='Run negative control test')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        exit(1)
    
    results = test_predictive_action(
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

