"""Core freedom computation engine for decision-preserving degrees of freedom."""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from core.rigidity import is_vit_model


@dataclass
class FreedomConfig:
    """Configuration for freedom computation."""
    
    # Perturbation parameters
    num_directions: int = 16  # Number of perturbation directions to sample
    epsilon_relative: float = 0.01  # Relative epsilon (1% of RMS of h)
    top_m_competitors: int = 10  # Number of competitor classes to track
    
    # Deterministic parameters
    deterministic_seed: int = 42
    
    # Probe layer selection
    probe_layers: Optional[List[str]] = None  # Auto-select if None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FreedomConfig':
        """Create from dictionary."""
        return cls(**d)


def get_probe_layers(model: nn.Module, is_vit: bool) -> List[str]:
    """
    Get list of probe layers for freedom computation.
    
    Args:
        model: PyTorch model
        is_vit: Whether model is a Vision Transformer
    
    Returns:
        List of layer names to probe (3-6 layers)
    """
    probe_layers = []
    
    if is_vit:
        # ViT: CLS output after a few encoder blocks + final LN output
        # Look for encoder blocks - ViT-B/16 has encoder.layers.encoder_layer_0 through encoder_layer_11
        # We want to probe at block OUTPUTS (not intermediate layers within blocks)
        # Format: encoder.layers.encoder_layer_N (the block module itself, not sub-modules)
        
        encoder_layers_list = []
        final_ln = None
        
        for name, module in model.named_modules():
            # Check for encoder layer blocks (the block module itself, not sub-modules)
            # Format: encoder.layers.encoder_layer_0, encoder.layers.encoder_layer_1, etc.
            if 'encoder.layers.encoder_layer_' in name:
                # Check if this is the block itself (not a sub-module)
                # Block name should be exactly "encoder.layers.encoder_layer_N" with no further dots
                parts = name.split('.')
                if len(parts) == 3:  # encoder.layers.encoder_layer_N (exactly 3 parts)
                    try:
                        block_num = int(parts[2].split('_')[-1])
                        encoder_layers_list.append((block_num, name))
                    except (ValueError, IndexError):
                        pass
            
            # Check for final layer norm (encoder.ln)
            if name == 'encoder.ln' or (name == 'ln' and hasattr(model, 'encoder')):
                final_ln = 'encoder.ln' if hasattr(model, 'encoder') else name
        
        # Sort by block number and select evenly spaced ones
        encoder_layers_list.sort(key=lambda x: x[0])
        
        if len(encoder_layers_list) >= 4:
            # Select blocks at indices: 2, 5, 8, 11 (or closest available)
            target_indices = [2, 5, 8, min(11, len(encoder_layers_list) - 1)]
            for idx in target_indices:
                if idx < len(encoder_layers_list):
                    probe_layers.append(encoder_layers_list[idx][1])
        else:
            # Take all available encoder layers
            for _, name in encoder_layers_list[:4]:
                probe_layers.append(name)
        
        # Add final LN if found
        if final_ln:
            probe_layers.append(final_ln)
        
        return probe_layers[:6] if len(probe_layers) > 6 else probe_layers
    
    else:
        # ResNet: after major blocks (layer1/2/3/4 outputs) + penultimate (avgpool)
        # Strategy: find the actual output layers (last relu in each block, or the block itself)
        # For ResNet-18, structure is: layer1 (2 blocks) -> layer2 (2 blocks) -> layer3 (2 blocks) -> layer4 (2 blocks) -> avgpool -> fc
        
        # Look for the actual layer modules (not sub-modules)
        layer_modules = {}
        for name, module in model.named_modules():
            if name == 'layer1':
                layer_modules['layer1'] = name
            elif name == 'layer2':
                layer_modules['layer2'] = name
            elif name == 'layer3':
                layer_modules['layer3'] = name
            elif name == 'layer4':
                layer_modules['layer4'] = name
            elif name == 'avgpool':
                layer_modules['avgpool'] = name
        
        # Also look for the last relu in each block (these are good probe points)
        # ResNet-18 has structure: layerX.0, layerX.1 (two basic blocks)
        # Each block ends with relu, so layerX.1.relu2 is the output
        last_relus = {}
        for name, module in model.named_modules():
            if name.endswith('.relu2') or name.endswith('.relu'):
                # Check which layer block this belongs to
                if '.1.relu' in name or name.endswith('.1.relu2'):
                    if 'layer1' in name and 'layer1' not in last_relus:
                        last_relus['layer1'] = name
                    elif 'layer2' in name and 'layer2' not in last_relus:
                        last_relus['layer2'] = name
                    elif 'layer3' in name and 'layer3' not in last_relus:
                        last_relus['layer3'] = name
                    elif 'layer4' in name and 'layer4' not in last_relus:
                        last_relus['layer4'] = name
        
        # Prefer last relus (actual outputs), fallback to layer modules
        for layer_key in ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            if layer_key in last_relus:
                probe_layers.append(last_relus[layer_key])
            elif layer_key in layer_modules:
                probe_layers.append(layer_modules[layer_key])
        
        return probe_layers[:6] if len(probe_layers) > 6 else probe_layers


def compute_epsilon(h: torch.Tensor, epsilon_relative: float) -> float:
    """
    Compute adaptive epsilon based on RMS of h.
    
    Args:
        h: Activation tensor
        epsilon_relative: Relative epsilon (e.g., 0.01 for 1%)
    
    Returns:
        Epsilon value: epsilon_relative * rms(h)
    """
    # Compute RMS over feature dimensions
    # For conv layers: [B, C, H, W] -> RMS over C, H, W (or just C)
    # For linear/ViT: [B, F] -> RMS over F
    
    if len(h.shape) == 4:
        # Conv: [B, C, H, W] -> flatten spatial, compute RMS per channel, then mean
        h_flat = h.flatten(2)  # [B, C, H*W]
        h_rms = torch.sqrt(torch.mean(h_flat ** 2, dim=2))  # [B, C]
        h_scale = float(torch.mean(h_rms).item())
    elif len(h.shape) == 3:
        # [B, T, F] or similar -> RMS over feature dim
        h_rms = torch.sqrt(torch.mean(h ** 2, dim=-1))  # [B, T] or [B, F]
        h_scale = float(torch.mean(h_rms).item())
    elif len(h.shape) == 2:
        # Linear/ViT CLS: [B, F] -> RMS over F
        h_rms = torch.sqrt(torch.mean(h ** 2, dim=-1))  # [B]
        h_scale = float(torch.mean(h_rms).item())
    elif len(h.shape) == 1:
        # [F] -> RMS
        h_scale = float(torch.sqrt(torch.mean(h ** 2)).item())
    else:
        # Fallback: RMS over all dimensions
        h_scale = float(torch.sqrt(torch.mean(h ** 2)).item())
    
    epsilon = epsilon_relative * h_scale
    return epsilon


def generate_perturbations(
    h: torch.Tensor,
    num_directions: int,
    epsilon: float,
    seed: int
) -> torch.Tensor:
    """
    Generate random Gaussian perturbation directions.
    
    Args:
        h: Activation tensor [B, ...] or [...]
        num_directions: Number of directions to generate
        epsilon: Perturbation magnitude (norm)
        seed: Random seed for determinism
    
    Returns:
        Perturbations tensor [num_directions, ...] with same shape as h (without batch dim)
    """
    # Set seed for determinism
    # Use same device as h
    rng = torch.Generator(device=h.device)
    rng.manual_seed(seed)
    
    # Get shape without batch dimension
    if len(h.shape) > 1 and h.shape[0] == 1:
        # Remove batch dimension for perturbation generation
        h_shape = h.shape[1:]
        h_flat_size = int(np.prod(h_shape))
    else:
        h_shape = h.shape
        h_flat_size = int(np.prod(h_shape))
    
    # Generate random directions in flattened space
    perturbations_flat = torch.randn(num_directions, h_flat_size, generator=rng, device=h.device, dtype=h.dtype)
    
    # Normalize each direction to unit norm
    norms = torch.norm(perturbations_flat, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)  # Avoid division by zero
    perturbations_flat = perturbations_flat / norms
    
    # Scale by epsilon
    perturbations_flat = perturbations_flat * epsilon
    
    # Reshape to match h shape (without batch)
    perturbations = perturbations_flat.reshape(num_directions, *h_shape)
    
    return perturbations


def create_tail_function(
    model: nn.Module,
    layer_name: str,
    original_input: torch.Tensor,
    competitor_set: List[int],
    is_vit: bool
) -> Callable:
    """
    Create a tail function that continues forward from a probe point.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to probe at
        original_input: Original input tensor to the model [B, ...]
        competitor_set: List of competitor class indices (for optimization)
        is_vit: Whether model is a Vision Transformer
    
    Returns:
        Function tail(h) -> competitor_logits that continues forward from probe point
        Returns only logits for competitor_set indices, not all classes
    
    Note: Uses handwritten tail functions for known architectures (ResNet, ViT).
    """
    if is_vit:
        return _create_vit_tail_function(model, layer_name, competitor_set)
    else:
        return _create_resnet_tail_function(model, layer_name, competitor_set)


def _create_resnet_tail_function(
    model: nn.Module,
    layer_name: str,
    competitor_set: List[int]
) -> Callable:
    """Create handwritten tail function for ResNet at specific splice points."""
    # Get model components
    layer1 = getattr(model, 'layer1', None)
    layer2 = getattr(model, 'layer2', None)
    layer3 = getattr(model, 'layer3', None)
    layer4 = getattr(model, 'layer4', None)
    avgpool = getattr(model, 'avgpool', None)
    fc = getattr(model, 'fc', None)
    
    if fc is None:
        raise ValueError("ResNet model missing 'fc' layer")
    
    def tail_fn(h: torch.Tensor) -> torch.Tensor:
        """
        Tail function for ResNet.
        
        Args:
            h: Activation tensor at probe point [B, C, H, W] or [B, C] or [B, C*H*W]
            For conv layers, h should be in feature map format [B, C, H, W]
            For avgpool, h should be flattened [B, C*H*W] or already pooled [B, C]
        
        Returns:
            Competitor logits [B, len(competitor_set)]
        """
        x = h
        
        # Determine which layers to run based on layer_name
        # Handle both "layer1" (module) and "layer1.1.relu2" (specific output) cases
        if 'layer1' in layer_name and 'layer2' not in layer_name:
            # Run layer2 → layer3 → layer4 → avgpool → fc
            if layer2 is not None:
                x = layer2(x)
            if layer3 is not None:
                x = layer3(x)
            if layer4 is not None:
                x = layer4(x)
        elif 'layer2' in layer_name and 'layer3' not in layer_name:
            # Run layer3 → layer4 → avgpool → fc
            if layer3 is not None:
                x = layer3(x)
            if layer4 is not None:
                x = layer4(x)
        elif 'layer3' in layer_name and 'layer4' not in layer_name:
            # Run layer4 → avgpool → fc
            if layer4 is not None:
                x = layer4(x)
        elif 'layer4' in layer_name:
            # Run avgpool → fc
            pass  # Already after layer4
        elif 'avgpool' in layer_name.lower():
            # Run fc only
            # h should already be flattened [B, C*H*W] or pooled [B, C]
            # If it's still [B, C, H, W], flatten it
            if len(x.shape) > 2:
                x = torch.flatten(x, 1)
        else:
            raise ValueError(f"Unknown ResNet splice point: {layer_name}")
        
        # Apply avgpool if not already done
        if avgpool is not None and 'avgpool' not in layer_name.lower():
            # x should be [B, C, H, W] at this point
            if len(x.shape) == 4:
                x = avgpool(x)
                x = torch.flatten(x, 1)
            else:
                # Already flattened or pooled
                pass
        
        # Apply fc (x should be [B, feature_dim] at this point)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        logits = fc(x)
        
        # Return only competitor logits
        competitor_logits = logits[:, competitor_set]
        return competitor_logits
    
    return tail_fn


def _create_vit_tail_function(
    model: nn.Module,
    layer_name: str,
    competitor_set: List[int]
) -> Callable:
    """Create handwritten tail function for ViT at specific splice points."""
    # Get model components
    encoder = getattr(model, 'encoder', None)
    ln = getattr(model, 'ln', None)  # Final layer norm
    heads = getattr(model, 'heads', None)  # Classification head
    
    if encoder is None or heads is None:
        raise ValueError("ViT model missing 'encoder' or 'heads' layer")
    
    # Find which encoder block we're probing at
    block_idx = None
    if 'encoder.layers.encoder_layer_' in layer_name:
        # Extract block index from layer name (e.g., "encoder.layers.encoder_layer_5" -> 5)
        try:
            parts = layer_name.split('.')
            if len(parts) >= 3:
                block_part = parts[2]  # encoder_layer_N
                block_idx = int(block_part.split('_')[-1])
        except (ValueError, IndexError):
            pass
    
    def tail_fn(h: torch.Tensor) -> torch.Tensor:
        """
        Tail function for ViT.
        
        Args:
            h: CLS token activation [B, D] at probe point
        
        Returns:
            Competitor logits [B, len(competitor_set)]
        """
        # h should be CLS token [B, D]
        if len(h.shape) != 2:
            raise ValueError(f"Expected CLS token [B, D], got shape {h.shape}")
        
        # Reconstruct full sequence with CLS token
        # For ViT, we need [B, 1, D] to match encoder input format
        # But actually, encoder expects [B, num_patches+1, D], so we need to handle this carefully
        # For v1, we'll assume h is already in the right format or we need to reconstruct
        
        # If we're probing at an encoder block output, we need to run remaining blocks
        # For ViT, encoder blocks expect [B, T, D] input (full token sequence)
        # But we're probing CLS-only [B, D], so we need to reconstruct the full sequence
        # For v1, we'll assume we can pass CLS as [B, 1, D] and it will work
        # (This is a simplification - in reality we'd need full tokens, but for CLS-only probing this should be acceptable)
        
        if block_idx is not None and encoder is not None:
            # Get encoder layers
            encoder_layers = encoder.layers if hasattr(encoder, 'layers') else encoder
            if isinstance(encoder_layers, nn.ModuleList) and block_idx + 1 < len(encoder_layers):
                # Reconstruct full sequence: h is CLS [B, D], we need [B, T, D]
                # For v1 simplification: use CLS as single token [B, 1, D]
                # This won't be perfectly accurate but should work for CLS-only probing
                x = h.unsqueeze(1)  # [B, 1, D]
                
                # Run remaining encoder blocks from block_idx+1 to end
                for i in range(block_idx + 1, len(encoder_layers)):
                    x = encoder_layers[i](x)
                
                # Extract CLS token (first token)
                if len(x.shape) == 3:
                    x = x[:, 0, :]  # [B, D]
            else:
                # No more blocks to run, h is already the output
                x = h
        elif 'encoder.ln' in layer_name or (layer_name == 'ln' and hasattr(model, 'encoder')):
            # Probing at final LN - h should be [B, D] (CLS token after final LN)
            # Don't apply ln again, just go to head
            x = h
        else:
            # Unknown probe point - try to continue anyway
            x = h
        
        # Apply final layer norm if present and we haven't already passed it
        if ln is not None and 'encoder.ln' not in layer_name and layer_name != 'ln':
            x = ln(x)
        
        # Apply classification head
        logits = heads(x)
        
        # Return only competitor logits
        competitor_logits = logits[:, competitor_set]
        return competitor_logits
    
    return tail_fn


def select_competitor_set(
    original_logits: torch.Tensor,
    top1: int,
    top_m: int
) -> List[int]:
    """
    Select competitor set for decision preservation test.
    
    Args:
        original_logits: Original logits tensor [B, num_classes] or [num_classes]
        top1: Index of top-1 predicted class
        top_m: Number of competitors to track (excluding top1)
    
    Returns:
        Sorted list of class indices: [top1, top2, ..., topM+1]
    """
    # Flatten to 1D if needed
    if len(original_logits.shape) > 1:
        logits_flat = original_logits[0]  # Take first batch item
    else:
        logits_flat = original_logits
    
    # Get top (top_m + 1) classes (including top1)
    _, top_indices = torch.topk(logits_flat, k=min(top_m + 1, len(logits_flat)))
    top_indices = top_indices.cpu().numpy().tolist()
    
    # Ensure top1 is first
    if top1 not in top_indices:
        top_indices.insert(0, top1)
        top_indices = top_indices[:top_m + 1]
    else:
        # Move top1 to front
        top_indices.remove(top1)
        top_indices.insert(0, top1)
    
    return top_indices


def test_decision_preservation_jvp(
    model: nn.Module,
    layer_name: str,
    h: torch.Tensor,
    perturbations: torch.Tensor,
    original_logits: torch.Tensor,
    original_top1: int,
    competitor_set: List[int],
    tail_fn: Callable
) -> torch.Tensor:
    """
    Test decision preservation using JVP (Jacobian-vector products).
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer being probed
        h: Original activation at layer_name [B, ...]
        perturbations: Perturbations [num_directions, ...] (without batch dim)
        original_logits: Original logits [B, num_classes] or [num_classes]
        original_top1: Index of original top-1 class
        competitor_set: List of competitor class indices
        tail_fn: Tail function tail(h) -> logits
    
    Returns:
        Boolean tensor [num_directions] indicating preservation
    """
    num_directions = perturbations.shape[0]
    preserved = torch.zeros(num_directions, dtype=torch.bool, device=h.device)
    
    # Flatten original logits if needed
    if len(original_logits.shape) > 1:
        original_logits_flat = original_logits[0]  # [num_classes]
    else:
        original_logits_flat = original_logits
    
    # For each perturbation, compute JVP and test preservation
    for i in range(num_directions):
        delta_h = perturbations[i]  # [...]
        
        # Add batch dimension if needed
        if len(delta_h.shape) < len(h.shape):
            delta_h = delta_h.unsqueeze(0)  # [1, ...]
        
        # Ensure delta_h has same shape as h
        if delta_h.shape != h.shape:
            # Broadcast or reshape as needed
            delta_h = delta_h.expand_as(h)
        
        # Special case: if delta_h is all zeros (epsilon=0), preservation is guaranteed
        if torch.allclose(delta_h, torch.zeros_like(delta_h), atol=1e-10):
            preserved[i] = True
            continue
        
        # Compute JVP: J @ delta_h where J is Jacobian of tail_fn at h
        # Use torch.autograd.functional.jvp
        try:
            # jvp computes: (f(x), J @ v) where f is tail_fn, x is h, v is delta_h
            _, jvp_result = torch.autograd.functional.jvp(
                tail_fn,
                h,
                delta_h
            )
            
            # jvp_result is competitor logits [B, len(competitor_set)] or [len(competitor_set)]
            # Take first batch item if needed
            if len(jvp_result.shape) > 1:
                jvp_competitor = jvp_result[0]  # [len(competitor_set)]
            else:
                jvp_competitor = jvp_result
            
            # Get original competitor logits
            original_competitor_logits = original_logits_flat[competitor_set]
            
            # Approximate perturbed competitor logits
            competitor_logits_perturbed = original_competitor_logits + jvp_competitor
            
            # Find index of top1 in competitor_set
            top1_idx_in_competitor = competitor_set.index(original_top1)
            logit_top1_pert = competitor_logits_perturbed[top1_idx_in_competitor]
            
            # Test preservation: logit_top1_pert > logit_k_pert for all k in competitor_set\{top1}
            preserved_flag = True
            
            for i_comp, k in enumerate(competitor_set):
                if k == original_top1:
                    continue
                if competitor_logits_perturbed[i_comp] >= logit_top1_pert:
                    preserved_flag = False
                    break
            
            preserved[i] = preserved_flag
            
        except Exception as e:
            # If JVP fails, mark as not preserved
            preserved[i] = False
    
    return preserved


def compute_brittleness_metrics(
    h: torch.Tensor,
    perturbations: torch.Tensor,
    original_logits: torch.Tensor,
    original_top1: int,
    competitor_set: List[int],
    tail_fn: Callable,
    epsilon: float
) -> Optional[Dict[str, float]]:
    """
    Compute brittleness metrics: distance to flip along margin-reducing directions.
    
    Implements Option A: compute epsilon_to_flip for each direction that reduces margin.
    
    Args:
        h: Original activation [B, ...]
        perturbations: Perturbations [num_directions, ...]
        original_logits: Original logits [B, num_classes] or [num_classes]
        original_top1: Index of original top-1 class
        competitor_set: List of competitor class indices
        tail_fn: Tail function
        epsilon: Epsilon used for perturbations (for normalization)
    
    Returns:
        Dict with brittleness metrics, or None if computation fails
    """
    try:
        # Flatten original logits if needed
        if len(original_logits.shape) > 1:
            original_logits_flat = original_logits[0]
        else:
            original_logits_flat = original_logits
        
        # Get original competitor logits from tail_fn(h) for consistency
        # Note: h should already have requires_grad=True when passed to this function
        # But we'll create a fresh one to be safe
        h_for_tail = h.clone().detach().requires_grad_(True)
        original_tail_logits = tail_fn(h_for_tail)[0].detach()
        original_competitor_logits = original_tail_logits
        
        # Compute margin gap: g = logit_top1 - max(logit_comp)
        top1_idx_in_competitor = competitor_set.index(original_top1)
        logit_top1_original = original_competitor_logits[top1_idx_in_competitor].item()
        
        # Find max competitor logit (excluding top1)
        competitor_logits_excluding_top1 = [
            original_competitor_logits[i].item()
            for i, class_idx in enumerate(competitor_set)
            if class_idx != original_top1
        ]
        max_competitor_logit = max(competitor_logits_excluding_top1) if competitor_logits_excluding_top1 else logit_top1_original
        
        margin_gap = logit_top1_original - max_competitor_logit
        
        if margin_gap <= 0:
            # Decision is already ambiguous, can't compute distance to flip
            return None
        
        # For each perturbation direction, compute margin change rate
        # Track delta_g for ALL directions (not just dangerous ones) for distribution metrics
        epsilon_to_flip_list = []
        dangerous_directions = 0
        delta_g_list = []  # Track all delta_g values for distribution metrics
        
        num_directions = perturbations.shape[0]
        
        for i in range(num_directions):
            delta_h = perturbations[i]
            
            # Add batch dimension if needed
            if len(delta_h.shape) < len(h.shape):
                delta_h = delta_h.unsqueeze(0)
            
            # Ensure delta_h has same shape as h
            if delta_h.shape != h.shape:
                delta_h = delta_h.expand_as(h)
            
            # Skip zero perturbations
            if torch.allclose(delta_h, torch.zeros_like(delta_h), atol=1e-10):
                continue
            
            try:
                # Compute JVP (need h with requires_grad=True)
                h_for_jvp = h.clone().detach().requires_grad_(True)
                _, jvp_result = torch.autograd.functional.jvp(tail_fn, h_for_jvp, delta_h)
                
                if len(jvp_result.shape) > 1:
                    jvp_competitor = jvp_result[0]
                else:
                    jvp_competitor = jvp_result
                
                # Compute margin change rate: Δg(v) = JVP_top1 - max(JVP_comp)
                jvp_top1 = jvp_competitor[top1_idx_in_competitor].item()
                
                # Find max JVP among competitors (excluding top1)
                jvp_competitors_excluding_top1 = [
                    jvp_competitor[j].item()
                    for j, class_idx in enumerate(competitor_set)
                    if class_idx != original_top1
                ]
                max_jvp_competitor = max(jvp_competitors_excluding_top1) if jvp_competitors_excluding_top1 else jvp_top1
                
                delta_g = jvp_top1 - max_jvp_competitor
                delta_g_list.append(delta_g)  # Track for distribution metrics
                
                # If direction reduces margin (Δg < 0), compute epsilon to flip
                if delta_g < 0:
                    dangerous_directions += 1
                    # Convention B: perturbations have norm = epsilon
                    # delta_g is the margin change for perturbation with norm = epsilon
                    # Formula: ε*(v) = ε * (g / (-Δg(δh)))
                    # where δh has norm = epsilon, and Δg(δh) is the margin change at that epsilon
                    if abs(delta_g) > 1e-10:  # Avoid division by zero
                        epsilon_to_flip = (margin_gap * epsilon) / (-delta_g)
                        epsilon_to_flip_list.append(epsilon_to_flip)
                    
            except Exception:
                # Skip this direction if JVP fails
                continue
        
        # Compute delta_margin distribution metrics (for ALL directions)
        if len(delta_g_list) > 0:
            delta_g_list_sorted = sorted(delta_g_list)
            delta_margin_min = float(delta_g_list_sorted[0])
            delta_margin_median = float(delta_g_list_sorted[len(delta_g_list_sorted) // 2])
            delta_margin_max = float(delta_g_list_sorted[-1])
        else:
            delta_margin_min = float('nan')
            delta_margin_median = float('nan')
            delta_margin_max = float('nan')
        
        # Build base result dict with metadata and distribution metrics
        result = {
            'scale_convention': 'epsilon_scaled_direction',  # Convention B: perturbations have norm = epsilon
            'epsilon_to_flip_units': 'absolute_epsilon',  # epsilon_to_flip values are in units of the epsilon used
            'delta_margin_min': delta_margin_min,
            'delta_margin_median': delta_margin_median,
            'delta_margin_max': delta_margin_max,
            'percent_dangerous_directions': float(dangerous_directions / num_directions) * 100.0,
            'margin_gap': margin_gap,
            'num_dangerous_directions': dangerous_directions
        }
        
        if len(epsilon_to_flip_list) == 0:
            # No dangerous directions found
            result.update({
                'epsilon_to_flip_min': float('inf'),
                'epsilon_to_flip_median': float('inf')
            })
        else:
            epsilon_to_flip_list.sort()
            result.update({
                'epsilon_to_flip_min': float(epsilon_to_flip_list[0]),
                'epsilon_to_flip_median': float(epsilon_to_flip_list[len(epsilon_to_flip_list) // 2])
            })
        
        return result
        
    except Exception as e:
        # If computation fails, return None
        return None


def compute_freedom_fraction(
    model: nn.Module,
    layer_name: str,
    h: torch.Tensor,
    config: FreedomConfig,
    original_logits: torch.Tensor,
    original_top1: int,
    tail_fn: Callable,
    is_vit: bool
) -> Tuple[float, List[int], float, int, Optional[Dict[str, float]]]:
    """
    Compute freedom fraction for a layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        h: Activation tensor at layer [B, ...]
        config: FreedomConfig
        original_logits: Original logits [B, num_classes]
        original_top1: Index of original top-1 class
        tail_fn: Tail function
        is_vit: Whether model is ViT
    
    Returns:
        Tuple of (freedom_fraction, competitor_indices, epsilon, probe_dim, brittleness_metrics)
        brittleness_metrics is None if computation fails, otherwise dict with:
        - epsilon_to_flip_min: minimum epsilon to flip along any direction
        - epsilon_to_flip_median: median epsilon to flip
        - percent_dangerous_directions: percentage of directions that reduce margin
    """
    # Compute probe dimension (dimensionality of h being perturbed)
    # For ResNet: if h is [B, C, H, W], we perturb in flattened feature space
    # For ViT: h is [B, D] (CLS token)
    if is_vit:
        # ViT CLS: [B, D] -> probe_dim = D
        probe_dim = int(h.shape[-1]) if len(h.shape) >= 2 else int(h.numel())
    else:
        # ResNet: perturb in feature space (flattened)
        # For conv layers, we typically pool or flatten before perturbing
        if len(h.shape) == 4:
            # [B, C, H, W] -> flatten to [B, C*H*W] or pool to [B, C]
            # For v1, we'll use the flattened size
            probe_dim = int(h.numel() // h.shape[0])  # Per-sample dimension
        else:
            probe_dim = int(h.numel() // h.shape[0]) if len(h.shape) > 1 else int(h.numel())
    
    # Compute adaptive epsilon
    epsilon = compute_epsilon(h, config.epsilon_relative)
    h_rms = float(torch.sqrt(torch.mean(h ** 2)).item()) if h.numel() > 0 else 0.0
    
    # Generate perturbations
    # Special case: if epsilon is exactly 0, all perturbations should be zero
    if epsilon == 0.0:
        # Create zero perturbations with correct shape
        if len(h.shape) > 1 and h.shape[0] == 1:
            h_shape = h.shape[1:]
        else:
            h_shape = h.shape
        perturbations = torch.zeros(config.num_directions, *h_shape, device=h.device, dtype=h.dtype)
    else:
        perturbations = generate_perturbations(
            h,
            config.num_directions,
            epsilon,
            config.deterministic_seed
        )
    
    # Select competitor set (needed for test)
    competitor_indices = select_competitor_set(
        original_logits,
        original_top1,
        config.top_m_competitors
    )
    
    # Test preservation
    preserved = test_decision_preservation_jvp(
        model,
        layer_name,
        h,
        perturbations,
        original_logits,
        original_top1,
        competitor_indices,
        tail_fn
    )
    
    # Compute freedom fraction
    num_preserved = int(torch.sum(preserved).item())
    freedom_fraction = float(num_preserved / config.num_directions)
    
    # Compute brittleness metrics (Option A: distance to flip)
    brittleness_metrics = compute_brittleness_metrics(
        h,
        perturbations,
        original_logits,
        original_top1,
        competitor_indices,
        tail_fn,
        epsilon
    )
    
    return freedom_fraction, competitor_indices, epsilon, probe_dim, brittleness_metrics


class FreedomEngine:
    """Engine for computing freedom maps from captured activations."""
    
    def __init__(self, model: nn.Module, config: Optional[FreedomConfig] = None, original_input: Optional[torch.Tensor] = None):
        """
        Initialize freedom engine.
        
        Args:
            model: PyTorch model
            config: FreedomConfig (uses defaults if None)
            original_input: Original input tensor used for forward pass (required for tail function)
        """
        self.model = model
        self.config = config or FreedomConfig()
        self.is_vit = is_vit_model(model)
        self.original_input = original_input
        
        if original_input is None:
            raise ValueError("original_input is required for tail function creation")
    
    def compute_freedom_map(
        self,
        activation_hooks: Dict[str, Any],
        logits: torch.Tensor,
        rigidity_map: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute freedom map from captured activations.
        
        Args:
            activation_hooks: Dict mapping layer_name -> ActivationHook
            logits: Final logits tensor [B, num_classes]
            rigidity_map: Optional rigidity map (for reference)
        
        Returns:
            Freedom map dictionary
        """
        import time
        start_time = time.time()
        
        # Get original top1
        if len(logits.shape) > 1:
            original_top1 = int(torch.argmax(logits, dim=1)[0].item())
            original_logits = logits
        else:
            original_top1 = int(torch.argmax(logits).item())
            original_logits = logits.unsqueeze(0)
        
        # Get probe layers
        if self.config.probe_layers is None:
            probe_layers = get_probe_layers(self.model, self.is_vit)
        else:
            probe_layers = self.config.probe_layers
        
        # Filter to only layers that have activations
        probe_layers = [layer for layer in probe_layers if layer in activation_hooks]
        
        if not probe_layers:
            raise ValueError("No probe layers found with activations")
        
        per_layer_freedom = {}
        
        # Process each probe layer
        for layer_name in probe_layers:
            hook = activation_hooks[layer_name]
            if not hook.activations:
                continue
            
            # Get activation h
            h = hook.activations[0]  # Use first activation (detached)
            
            # Handle ViT blocks: extract CLS token if needed
            if self.is_vit:
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        module_str = str(type(module)).lower()
                        is_vit_block = 'encoderblock' in module_str or 'transformerblock' in module_str
                        if is_vit_block and len(h.shape) == 3:
                            # Extract CLS token [B, T, F] -> [B, F]
                            h = h[:, 0, :]
                        break
            
            # Clone h and enable gradients for JVP
            # h is detached from hooks, so we clone it and set requires_grad
            h_grad = h.clone().detach().requires_grad_(True)
            
            # Select competitor set first (needed for tail function)
            competitor_indices = select_competitor_set(
                original_logits,
                original_top1,
                self.config.top_m_competitors
            )
            
            # Create tail function for this layer (with competitor set baked in)
            tail_fn = create_tail_function(
                self.model,
                layer_name,
                self.original_input,
                competitor_indices,
                self.is_vit
            )
            
            # Compute freedom fraction
            try:
                freedom_fraction, competitor_indices, epsilon, probe_dim, brittleness_metrics = compute_freedom_fraction(
                    self.model,
                    layer_name,
                    h_grad,  # Use h with gradients enabled
                    self.config,
                    original_logits,
                    original_top1,
                    tail_fn,
                    self.is_vit
                )
                
                # Compute h_rms for logging (use original h, not h_grad)
                h_rms = float(torch.sqrt(torch.mean(h ** 2)).item()) if h.numel() > 0 else 0.0
                
                per_layer_freedom[layer_name] = {
                    'freedom_fraction': freedom_fraction,
                    'num_directions': self.config.num_directions,
                    'epsilon': epsilon,
                    'epsilon_relative': self.config.epsilon_relative,
                    'h_rms': h_rms,
                    'probe_dim': probe_dim,
                    'seed': self.config.deterministic_seed,
                    'competitor_indices': competitor_indices
                }
                
                # Add brittleness metrics if available
                if brittleness_metrics is not None:
                    per_layer_freedom[layer_name].update(brittleness_metrics)
                    
            except Exception as e:
                # If computation fails for a layer, skip it
                print(f"Warning: Failed to compute freedom for layer {layer_name}: {e}")
                continue
        
        # Compute global stability metrics
        if per_layer_freedom:
            freedom_fractions = [v['freedom_fraction'] for v in per_layer_freedom.values()]
            mean_freedom = float(np.mean(freedom_fractions))
            min_freedom = float(np.min(freedom_fractions))
            
            # Find bottleneck layer (lowest freedom)
            bottleneck_layer = min(per_layer_freedom.items(), key=lambda x: x[1]['freedom_fraction'])[0]
        else:
            mean_freedom = 0.0
            min_freedom = 0.0
            bottleneck_layer = None
        
        # Compute original top1 probability
        probs = torch.softmax(original_logits, dim=1)
        original_top1_prob = float(probs[0, original_top1].item())
        
        runtime_ms = (time.time() - start_time) * 1000
        
        # Build freedom map
        freedom_map = {
            'config': self.config.to_dict(),
            'per_layer_freedom': per_layer_freedom,
            'global_stability': {
                'mean_freedom': mean_freedom,
                'min_freedom': min_freedom,
                'freedom_bottleneck_layer': bottleneck_layer
            },
            'metadata': {
                'original_top1': original_top1,
                'original_top1_prob': original_top1_prob,
                'runtime_ms': runtime_ms,
                'num_probe_layers': len(probe_layers),
                'num_layers_computed': len(per_layer_freedom)
            }
        }
        
        return freedom_map

