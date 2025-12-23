"""Test suite for Phase 2 (Conditional Freedom) certification."""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import numpy as np

from core.instrumentation import ActivationCapture
from core.freedom import FreedomEngine, FreedomConfig
from core.rigidity import RigidityEngine, RigidityConfig
from demos.compute_rigidity import load_test_image


def validate_freedom_schema(freedom_map: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate freedom map schema and ranges.
    
    Args:
        freedom_map: Freedom map dictionary
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['config', 'per_layer_freedom', 'global_stability', 'metadata']
    for key in required_keys:
        if key not in freedom_map:
            errors.append(f"Missing required key: {key}")
    
    # Check config
    if 'config' in freedom_map:
        config = freedom_map['config']
        required_config_keys = ['num_directions', 'epsilon_relative', 'top_m_competitors', 'deterministic_seed']
        for key in required_config_keys:
            if key not in config:
                errors.append(f"Missing config key: {key}")
    
    # Check per_layer_freedom
    if 'per_layer_freedom' in freedom_map:
        for layer_name, layer_data in freedom_map['per_layer_freedom'].items():
            required_layer_keys = [
                'freedom_fraction', 'num_directions', 'epsilon', 'epsilon_relative',
                'h_rms', 'seed', 'competitor_indices'
            ]
            for key in required_layer_keys:
                if key not in layer_data:
                    errors.append(f"Missing key in per_layer_freedom[{layer_name}]: {key}")
            
            # Check ranges
            if 'freedom_fraction' in layer_data:
                ff = layer_data['freedom_fraction']
                if not isinstance(ff, (int, float)) or ff < 0.0 or ff > 1.0:
                    errors.append(f"freedom_fraction out of range [0,1] in {layer_name}: {ff}")
            
            if 'competitor_indices' in layer_data:
                comp_indices = layer_data['competitor_indices']
                if not isinstance(comp_indices, list) or len(comp_indices) == 0:
                    errors.append(f"competitor_indices must be non-empty list in {layer_name}")
            
            # Check brittleness metrics if present (optional)
            if 'epsilon_to_flip_min' in layer_data:
                # If brittleness metrics exist, validate all required fields
                brittleness_keys = [
                    'epsilon_to_flip_min', 'epsilon_to_flip_median', 'percent_dangerous_directions',
                    'margin_gap', 'scale_convention', 'epsilon_to_flip_units',
                    'delta_margin_min', 'delta_margin_median', 'delta_margin_max'
                ]
                for key in brittleness_keys:
                    if key not in layer_data:
                        errors.append(f"Missing brittleness metric key in per_layer_freedom[{layer_name}]: {key}")
                
                # Validate scale_convention
                if 'scale_convention' in layer_data:
                    sc = layer_data['scale_convention']
                    if sc not in ['epsilon_scaled_direction', 'unit_direction']:
                        errors.append(f"scale_convention must be 'epsilon_scaled_direction' or 'unit_direction' in {layer_name}: {sc}")
                
                # Validate epsilon_to_flip_units
                if 'epsilon_to_flip_units' in layer_data:
                    etfu = layer_data['epsilon_to_flip_units']
                    if not isinstance(etfu, str):
                        errors.append(f"epsilon_to_flip_units must be string in {layer_name}: {etfu}")
                
                # Validate numeric ranges
                if 'epsilon_to_flip_min' in layer_data:
                    etf_min = layer_data['epsilon_to_flip_min']
                    if not isinstance(etf_min, (int, float)) or (etf_min != float('inf') and etf_min < 0):
                        errors.append(f"epsilon_to_flip_min must be non-negative or inf in {layer_name}: {etf_min}")
                
                if 'percent_dangerous_directions' in layer_data:
                    pdd = layer_data['percent_dangerous_directions']
                    if not isinstance(pdd, (int, float)) or pdd < 0.0 or pdd > 100.0:
                        errors.append(f"percent_dangerous_directions out of range [0,100] in {layer_name}: {pdd}")
            
            # MANDATORY: If epsilon_relative > 0, brittleness metrics MUST exist
            if 'config' in freedom_map:
                config = freedom_map['config']
                epsilon_relative = config.get('epsilon_relative', 0.0)
                if epsilon_relative > 0:
                    # Check that brittleness metrics exist
                    if 'epsilon_to_flip_min' not in layer_data:
                        errors.append(f"MANDATORY: brittleness metrics missing in {layer_name} (epsilon_relative={epsilon_relative} > 0)")
                    else:
                        # Verify all required fields exist
                        required_brittleness_fields = [
                            'scale_convention', 'epsilon_to_flip_units',
                            'delta_margin_min', 'delta_margin_median', 'delta_margin_max',
                            'margin_gap', 'percent_dangerous_directions'
                        ]
                        for field in required_brittleness_fields:
                            if field not in layer_data:
                                errors.append(f"MANDATORY: brittleness field '{field}' missing in {layer_name} (epsilon_relative > 0)")
                        
                        # Verify at least one epsilon_to_flip metric is valid OR percent_dangerous_directions == 0
                        etf_min = layer_data.get('epsilon_to_flip_min', None)
                        etf_median = layer_data.get('epsilon_to_flip_median', None)
                        pdd = layer_data.get('percent_dangerous_directions', None)
                        
                        has_valid_etf = (
                            (etf_min is not None and etf_min != float('inf')) or
                            (etf_median is not None and etf_median != float('inf'))
                        )
                        is_no_danger = (pdd is not None and pdd == 0.0)
                        
                        if not (has_valid_etf or is_no_danger):
                            errors.append(
                                f"MANDATORY: brittleness metrics invalid in {layer_name}: "
                                f"epsilon_to_flip_min={etf_min}, epsilon_to_flip_median={etf_median}, "
                                f"percent_dangerous_directions={pdd} (must have valid epsilon_to_flip OR pdd==0)"
                            )
    
    # Check global_stability
    if 'global_stability' in freedom_map:
        gs = freedom_map['global_stability']
        required_gs_keys = ['mean_freedom', 'min_freedom', 'freedom_bottleneck_layer']
        for key in required_gs_keys:
            if key not in gs:
                errors.append(f"Missing global_stability key: {key}")
        
        if 'mean_freedom' in gs:
            mf = gs['mean_freedom']
            if not isinstance(mf, (int, float)) or mf < 0.0 or mf > 1.0:
                errors.append(f"mean_freedom out of range [0,1]: {mf}")
        
        if 'min_freedom' in gs:
            minf = gs['min_freedom']
            if not isinstance(minf, (int, float)) or minf < 0.0 or minf > 1.0:
                errors.append(f"min_freedom out of range [0,1]: {minf}")
    
    return len(errors) == 0, errors


def validate_brittleness_metrics_in_golden(freedom_map: Dict[str, Any], golden_map: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """
    Validate brittleness metrics consistency with golden baseline.
    
    Args:
        freedom_map: Current freedom map
        golden_map: Optional golden baseline freedom map
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if 'per_layer_freedom' not in freedom_map:
        return False, ['per_layer_freedom missing in freedom_map']
    
    for layer_name, layer_data in freedom_map['per_layer_freedom'].items():
        # Check scale_convention is present and unchanged
        if 'scale_convention' not in layer_data:
            errors.append(f"scale_convention missing in {layer_name}")
        else:
            sc = layer_data['scale_convention']
            if sc not in ['epsilon_scaled_direction', 'unit_direction']:
                errors.append(f"Invalid scale_convention in {layer_name}: {sc}")
            
            # If golden exists, check it matches
            if golden_map and 'per_layer_freedom' in golden_map:
                if layer_name in golden_map['per_layer_freedom']:
                    golden_sc = golden_map['per_layer_freedom'][layer_name].get('scale_convention')
                    if golden_sc and golden_sc != sc:
                        errors.append(f"scale_convention changed in {layer_name}: {golden_sc} -> {sc}")
        
        # Check epsilon_to_flip_units is present and unchanged
        if 'epsilon_to_flip_units' not in layer_data:
            errors.append(f"epsilon_to_flip_units missing in {layer_name}")
        else:
            etfu = layer_data['epsilon_to_flip_units']
            if not isinstance(etfu, str):
                errors.append(f"epsilon_to_flip_units must be string in {layer_name}: {etfu}")
            
            # If golden exists, check it matches
            if golden_map and 'per_layer_freedom' in golden_map:
                if layer_name in golden_map['per_layer_freedom']:
                    golden_etfu = golden_map['per_layer_freedom'][layer_name].get('epsilon_to_flip_units')
                    if golden_etfu and golden_etfu != etfu:
                        errors.append(f"epsilon_to_flip_units changed in {layer_name}: {golden_etfu} -> {etfu}")
        
        # Check delta_margin fields are present and numeric
        for field in ['delta_margin_min', 'delta_margin_median', 'delta_margin_max']:
            if field not in layer_data:
                errors.append(f"{field} missing in {layer_name}")
            else:
                val = layer_data[field]
                if not isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                    errors.append(f"{field} must be numeric in {layer_name}: {val}")
        
        # Check epsilon_to_flip_min stability (if golden exists)
        if golden_map and 'per_layer_freedom' in golden_map:
            if layer_name in golden_map['per_layer_freedom']:
                golden_layer = golden_map['per_layer_freedom'][layer_name]
                if 'epsilon_to_flip_min' in layer_data and 'epsilon_to_flip_min' in golden_layer:
                    current_etf = layer_data['epsilon_to_flip_min']
                    golden_etf = golden_layer['epsilon_to_flip_min']
                    current_pdd = layer_data.get('percent_dangerous_directions', None)
                    golden_pdd = golden_layer.get('percent_dangerous_directions', None)
                    
                    # Handle None/inf cases properly
                    current_is_inf = (current_etf == float('inf') or current_etf is None)
                    golden_is_inf = (golden_etf == float('inf') or golden_etf is None)
                    
                    if golden_is_inf:
                        # Golden had no dangerous directions
                        if not current_is_inf:
                            # Current has dangerous directions - this is a change
                            errors.append(
                                f"epsilon_to_flip_min changed from no-danger to dangerous in {layer_name}: "
                                f"golden=None/inf, current={current_etf:.6f}"
                            )
                        # If both are inf, that's fine - check percent_dangerous_directions consistency
                        if current_pdd is not None and golden_pdd is not None:
                            if golden_pdd == 0.0 and current_pdd != 0.0:
                                errors.append(
                                    f"percent_dangerous_directions changed from 0% in {layer_name}: "
                                    f"golden={golden_pdd}%, current={current_pdd}%"
                                )
                    elif current_is_inf:
                        # Golden had dangerous directions, current doesn't - this is a change
                        errors.append(
                            f"epsilon_to_flip_min changed from dangerous to no-danger in {layer_name}: "
                            f"golden={golden_etf:.6f}, current=None/inf"
                        )
                    else:
                        # Both are finite - check stability
                        if golden_etf > 0:
                            relative_diff = abs(current_etf - golden_etf) / golden_etf
                            tolerance = 0.15  # 15% tolerance for golden comparison
                            if relative_diff > tolerance:
                                errors.append(
                                    f"epsilon_to_flip_min changed significantly in {layer_name}: "
                                    f"{golden_etf:.6f} -> {current_etf:.6f} (diff: {relative_diff*100:.2f}%)"
                                )
    
    return len(errors) == 0, errors


def test_a_schema_ranges(image_path: Path, model_name: str) -> Dict[str, Any]:
    """Test A: Schema and ranges validation."""
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
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    # Capture activations
    capture = ActivationCapture(model, layer_names=None)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    logits = output[0] if isinstance(output, tuple) else output
    
    # Compute freedom map
    freedom_config = FreedomConfig()
    freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
    
    try:
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
        
        # Validate schema
        is_valid, errors = validate_freedom_schema(freedom_map)
        
        if not is_valid:
            return {
                'status': 'FAIL',
                'details': {
                    'schema_valid': False
                },
                'errors': errors
            }
        
        # Validate brittleness metrics consistency (check against golden if available)
        golden_path = Path('eval/golden') / model_name / 'freedom_map.json'
        golden_map = None
        if golden_path.exists():
            try:
                with open(golden_path, 'r') as f:
                    golden_map = json.load(f)
            except Exception:
                pass  # Skip golden validation if file can't be loaded
        
        brittleness_valid, brittleness_errors = validate_brittleness_metrics_in_golden(freedom_map, golden_map)
        if not brittleness_valid:
            errors.extend(brittleness_errors)
            return {
                'status': 'FAIL',
                'details': {
                    'schema_valid': True,
                    'brittleness_valid': False,
                    'golden_available': golden_map is not None
                },
                'errors': brittleness_errors
            }
        
        return {
            'status': 'PASS',
            'details': {
                'num_layers': len(freedom_map['per_layer_freedom']),
                'schema_valid': True,
                'brittleness_valid': True,
                'golden_available': golden_map is not None
            },
            'errors': []
        }
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f'Exception during freedom computation: {str(e)}']
        }
    finally:
        capture.remove_hooks()


def test_b_determinism(image_path: Path, model_name: str) -> Dict[str, Any]:
    """Test B: Determinism (3-run stability)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tolerance = 1e-6
    
    # Load model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
    else:
        return {'status': 'FAIL', 'errors': [f'Unknown model: {model_name}']}
    
    model = model.to(device)
    model.eval()
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    freedom_maps = []
    
    for run in range(3):
        # Capture activations
        capture = ActivationCapture(model, layer_names=None)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        logits = output[0] if isinstance(output, tuple) else output
        
        # Compute freedom map
        freedom_config = FreedomConfig()
        freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
        
        try:
            freedom_map = freedom_engine.compute_freedom_map(
                activation_hooks=capture.activation_hooks,
                logits=logits
            )
            freedom_maps.append(freedom_map)
        except Exception as e:
            return {
                'status': 'FAIL',
                'errors': [f'Exception in run {run+1}: {str(e)}']
            }
        finally:
            capture.remove_hooks()
    
    # Compare runs
    if len(freedom_maps) < 3:
        return {
            'status': 'FAIL',
            'errors': ['Failed to generate 3 freedom maps']
        }
    
    # Compare freedom fractions per layer
    max_diff = 0.0
    all_layers = set()
    for fm in freedom_maps:
        all_layers.update(fm['per_layer_freedom'].keys())
    
    for layer_name in all_layers:
        fractions = []
        for fm in freedom_maps:
            if layer_name in fm['per_layer_freedom']:
                fractions.append(fm['per_layer_freedom'][layer_name]['freedom_fraction'])
        
        if len(fractions) == 3:
            diff = max(fractions) - min(fractions)
            max_diff = max(max_diff, diff)
    
    if max_diff > tolerance:
        return {
            'status': 'FAIL',
            'details': {
                'max_per_layer_diff': max_diff,
                'tolerance': tolerance
            },
            'errors': [f'Max per-layer difference {max_diff} exceeds tolerance {tolerance}']
        }
    else:
        return {
            'status': 'PASS',
            'details': {
                'max_per_layer_diff': max_diff,
                'tolerance': tolerance
            },
            'errors': []
        }


def test_c_monotone_stress(image_path: Path, model_name: str) -> Dict[str, Any]:
    """Test C: Monotone stress check - corruption should NOT increase freedom at ALL layers simultaneously."""
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
    
    # Load clean image
    input_tensor_clean = load_test_image(image_path, device)
    
    # Create corrupted image (add noise)
    input_tensor_corrupted = input_tensor_clean.clone()
    noise = torch.randn_like(input_tensor_corrupted) * 0.1
    input_tensor_corrupted = input_tensor_corrupted + noise
    input_tensor_corrupted = torch.clamp(input_tensor_corrupted, 0, 1)
    
    def compute_freedom(input_tensor):
        capture = ActivationCapture(model, layer_names=None)
        with torch.no_grad():
            output = model(input_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        
        freedom_config = FreedomConfig()
        freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
        capture.remove_hooks()
        return freedom_map
    
    try:
        freedom_clean = compute_freedom(input_tensor_clean)
        freedom_corrupted = compute_freedom(input_tensor_corrupted)
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f'Exception during freedom computation: {str(e)}']
        }
    
    # Check: corruption should NOT increase freedom at ALL layers simultaneously
    # Find common layers
    clean_layers = set(freedom_clean['per_layer_freedom'].keys())
    corrupted_layers = set(freedom_corrupted['per_layer_freedom'].keys())
    common_layers = clean_layers & corrupted_layers
    
    if not common_layers:
        return {
            'status': 'FAIL',
            'errors': ['No common layers between clean and corrupted runs']
        }
    
    # Count layers where freedom increased
    increased_count = 0
    for layer_name in common_layers:
        ff_clean = freedom_clean['per_layer_freedom'][layer_name]['freedom_fraction']
        ff_corrupted = freedom_corrupted['per_layer_freedom'][layer_name]['freedom_fraction']
        if ff_corrupted > ff_clean:
            increased_count += 1
    
    # If ALL layers increased, that's a failure
    if increased_count == len(common_layers):
        return {
            'status': 'FAIL',
            'details': {
                'common_layers': len(common_layers),
                'increased_count': increased_count
            },
            'errors': ['Corruption increased freedom at ALL layers simultaneously']
        }
    else:
        return {
            'status': 'PASS',
            'details': {
                'common_layers': len(common_layers),
                'increased_count': increased_count,
                'decreased_or_same_count': len(common_layers) - increased_count
            },
            'errors': []
        }


def test_d_runtime(image_path: Path, model_name: str) -> Dict[str, Any]:
    """Test D: Runtime < 1s on RTX 2060 Super (measure forward + freedom_compute separately)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_runtime_s = 1.0  # 1 second
    
    # Load model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
    else:
        return {'status': 'FAIL', 'errors': [f'Unknown model: {model_name}']}
    
    model = model.to(device)
    model.eval()
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    # Measure forward pass
    capture = ActivationCapture(model, layer_names=None)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    forward_time = t1 - t0
    
    logits = output[0] if isinstance(output, tuple) else output
    
    # Measure freedom computation
    freedom_config = FreedomConfig()
    freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    try:
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f'Exception during freedom computation: {str(e)}']
        }
    finally:
        capture.remove_hooks()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    freedom_time = t1 - t0
    
    total_time = forward_time + freedom_time
    
    if total_time > max_runtime_s:
        return {
            'status': 'FAIL',
            'details': {
                'forward_time': forward_time,
                'freedom_time': freedom_time,
                'total_time': total_time,
                'max_runtime': max_runtime_s
            },
            'errors': [f'Total runtime {total_time:.3f}s exceeds max {max_runtime_s}s']
        }
    else:
        return {
            'status': 'PASS',
            'details': {
                'forward_time': forward_time,
                'freedom_time': freedom_time,
                'total_time': total_time,
                'max_runtime': max_runtime_s
            },
            'errors': []
        }


def test_e_epsilon_zero_baseline(image_path: Path, model_name: str) -> Dict[str, Any]:
    """Test E: If epsilon_relative = 0, all freedom fractions must be 1.0."""
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
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    # Capture activations
    capture = ActivationCapture(model, layer_names=None)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    logits = output[0] if isinstance(output, tuple) else output
    
    # Compute freedom map with epsilon_relative = 0
    freedom_config = FreedomConfig(epsilon_relative=0.0)
    freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
    
    try:
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
        
        # Check: all freedom fractions should be 1.0 (within tolerance)
        tolerance = 1e-6
        all_one = True
        failed_layers = []
        
        for layer_name, layer_data in freedom_map['per_layer_freedom'].items():
            ff = layer_data['freedom_fraction']
            if abs(ff - 1.0) > tolerance:
                all_one = False
                failed_layers.append(f"{layer_name}: {ff} (expected 1.0)")
        
        if not all_one:
            return {
                'status': 'FAIL',
                'details': {
                    'failed_layers': failed_layers,
                    'tolerance': tolerance
                },
                'errors': [f'With epsilon_relative=0, some layers have freedom != 1.0: {failed_layers}']
            }
        else:
            return {
                'status': 'PASS',
                'details': {
                    'num_layers': len(freedom_map['per_layer_freedom']),
                    'all_freedom_fractions': 1.0
                },
                'errors': []
            }
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f'Exception during freedom computation: {str(e)}']
        }
    finally:
        capture.remove_hooks()


def test_f_epsilon_scaling_sanity(image_path: Path, model_name: str) -> Dict[str, Any]:
    """
    Test F: Epsilon scaling sanity check - validates Convention B consistency.
    
    Checks:
    1. epsilon_to_flip_min should remain stable when epsilon_relative changes (within 10%)
    2. delta_margin_* should scale by ~2× when epsilon doubles (proves delta_g is computed at epsilon)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tolerance_etf = 0.10  # 10% tolerance for epsilon_to_flip_min stability
    tolerance_delta = 0.20  # 20% tolerance for delta_margin scaling (2× expected)
    
    # Load model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
    else:
        return {'status': 'FAIL', 'errors': [f'Unknown model: {model_name}']}
    
    model = model.to(device)
    model.eval()
    
    # Load image
    input_tensor = load_test_image(image_path, device)
    
    def compute_freedom_with_epsilon(epsilon_rel: float):
        """Helper to compute freedom map with given epsilon_relative."""
        capture = ActivationCapture(model, layer_names=None)
        with torch.no_grad():
            output = model(input_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        
        freedom_config = FreedomConfig(epsilon_relative=epsilon_rel)
        freedom_engine = FreedomEngine(model, freedom_config, original_input=input_tensor)
        freedom_map = freedom_engine.compute_freedom_map(
            activation_hooks=capture.activation_hooks,
            logits=logits
        )
        capture.remove_hooks()
        return freedom_map
    
    try:
        # Run with epsilon_relative = 0.01
        freedom_map_1 = compute_freedom_with_epsilon(0.01)
        
        # Run with epsilon_relative = 0.02
        freedom_map_2 = compute_freedom_with_epsilon(0.02)
        
        # Compare epsilon_to_flip_min for each layer
        layers_1 = set(freedom_map_1['per_layer_freedom'].keys())
        layers_2 = set(freedom_map_2['per_layer_freedom'].keys())
        common_layers = layers_1 & layers_2
        
        if not common_layers:
            return {
                'status': 'FAIL',
                'errors': ['No common layers between epsilon_relative=0.01 and 0.02 runs']
            }
        
        max_relative_diff_etf = 0.0
        max_relative_diff_delta = 0.0
        max_abs_diff_pdd = 0.0
        failed_layers_etf = []
        failed_layers_delta = []
        failed_layers_pdd = []
        
        for layer_name in common_layers:
            layer_data_1 = freedom_map_1['per_layer_freedom'][layer_name]
            layer_data_2 = freedom_map_2['per_layer_freedom'][layer_name]
            
            # Check if brittleness metrics exist
            if 'epsilon_to_flip_min' not in layer_data_1 or 'epsilon_to_flip_min' not in layer_data_2:
                continue  # Skip layers without brittleness metrics
            
            # Check 1: epsilon_to_flip_min stability
            etf_min_1 = layer_data_1['epsilon_to_flip_min']
            etf_min_2 = layer_data_2['epsilon_to_flip_min']
            
            # Skip if either is inf (no dangerous directions)
            if etf_min_1 != float('inf') and etf_min_2 != float('inf'):
                if etf_min_1 > 0:
                    relative_diff = abs(etf_min_2 - etf_min_1) / etf_min_1
                    max_relative_diff_etf = max(max_relative_diff_etf, relative_diff)
                    
                    if relative_diff > tolerance_etf:
                        failed_layers_etf.append(f"{layer_name}: {etf_min_1:.6f} -> {etf_min_2:.6f} (diff: {relative_diff*100:.2f}%)")
            
            # Check 2: delta_margin scaling (should scale by ~2× when epsilon doubles)
            for delta_field in ['delta_margin_min', 'delta_margin_median', 'delta_margin_max']:
                if delta_field in layer_data_1 and delta_field in layer_data_2:
                    delta_1 = layer_data_1[delta_field]
                    delta_2 = layer_data_2[delta_field]
                    
                    # Skip NaN values
                    if isinstance(delta_1, float) and np.isnan(delta_1):
                        continue
                    if isinstance(delta_2, float) and np.isnan(delta_2):
                        continue
                    
                    # Skip zero values
                    if abs(delta_1) < 1e-10:
                        continue
                    
                    # Expected: delta_2 ≈ 2 * delta_1 (since epsilon doubled)
                    expected_delta_2 = 2.0 * delta_1
                    actual_ratio = delta_2 / delta_1 if abs(delta_1) > 1e-10 else 0.0
                    expected_ratio = 2.0
                    
                    # Check if ratio is close to 2.0
                    ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio
                    max_relative_diff_delta = max(max_relative_diff_delta, ratio_diff)
                    
                    if ratio_diff > tolerance_delta:
                        failed_layers_delta.append(
                            f"{layer_name}.{delta_field}: {delta_1:.6f} -> {delta_2:.6f} "
                            f"(ratio: {actual_ratio:.3f}, expected: {expected_ratio:.3f}, diff: {ratio_diff*100:.2f}%)"
                        )
            
            # Check 3: percent_dangerous_directions stability (should be roughly constant)
            if 'percent_dangerous_directions' in layer_data_1 and 'percent_dangerous_directions' in layer_data_2:
                pdd_1 = layer_data_1['percent_dangerous_directions']
                pdd_2 = layer_data_2['percent_dangerous_directions']
                
                # Check absolute difference (within 20% absolute)
                pdd_diff = abs(pdd_2 - pdd_1)
                max_abs_diff_pdd = max(max_abs_diff_pdd, pdd_diff)
                
                if pdd_diff > 20.0:  # 20% absolute tolerance
                    failed_layers_pdd.append(
                        f"{layer_name}: {pdd_1:.2f}% -> {pdd_2:.2f}% (diff: {pdd_diff:.2f}%)"
                    )
        
        # Combine failures
        all_failed = failed_layers_etf + failed_layers_delta + failed_layers_pdd
        max_failure = max(max_relative_diff_etf, max_relative_diff_delta)
        
        if max_relative_diff_etf > tolerance_etf or max_relative_diff_delta > tolerance_delta or max_abs_diff_pdd > 20.0:
            return {
                'status': 'FAIL',
                'details': {
                    'max_relative_diff_etf': max_relative_diff_etf,
                    'max_relative_diff_delta': max_relative_diff_delta,
                    'max_abs_diff_pdd': max_abs_diff_pdd,
                    'tolerance_etf': tolerance_etf,
                    'tolerance_delta': tolerance_delta,
                    'tolerance_pdd': 20.0,
                    'failed_layers_etf': failed_layers_etf,
                    'failed_layers_delta': failed_layers_delta,
                    'failed_layers_pdd': failed_layers_pdd
                },
                'errors': [
                    err for err in [
                        f'epsilon_to_flip_min changed by more than {tolerance_etf*100:.0f}% tolerance: max_diff={max_relative_diff_etf*100:.2f}%' if max_relative_diff_etf > tolerance_etf else None,
                        f'delta_margin scaling failed (expected ~2× when epsilon doubles): max_diff={max_relative_diff_delta*100:.2f}%' if max_relative_diff_delta > tolerance_delta else None,
                        f'percent_dangerous_directions changed by more than 20% absolute: max_diff={max_abs_diff_pdd:.2f}%' if max_abs_diff_pdd > 20.0 else None
                    ] if err is not None
                ]
            }
        else:
            return {
                'status': 'PASS',
                'details': {
                    'max_relative_diff_etf': max_relative_diff_etf,
                    'max_relative_diff_delta': max_relative_diff_delta,
                    'max_abs_diff_pdd': max_abs_diff_pdd,
                    'tolerance_etf': tolerance_etf,
                    'tolerance_delta': tolerance_delta,
                    'tolerance_pdd': 20.0,
                    'num_layers_compared': len([l for l in common_layers 
                                                if 'epsilon_to_flip_min' in freedom_map_1['per_layer_freedom'][l] 
                                                and 'epsilon_to_flip_min' in freedom_map_2['per_layer_freedom'][l]
                                                and freedom_map_1['per_layer_freedom'][l]['epsilon_to_flip_min'] != float('inf')
                                                and freedom_map_2['per_layer_freedom'][l]['epsilon_to_flip_min'] != float('inf')])
                },
                'errors': []
            }
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f'Exception during epsilon scaling sanity check: {str(e)}']
        }

