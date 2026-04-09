"""
Utilities for hashing input regions for cache keys.

Provides deterministic hashing of regions to enable caching of bound
propagation results.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from bound_propagation.regions.abstract import AbstractRegion
    from bound_propagation.regions.hyperrectangle import HyperRectangle
    from bound_propagation.regions.multi_input import MultiInputRegion


def hash_tensor(tensor: torch.Tensor) -> str:
    """
    Compute a deterministic hash of a tensor.
    
    Args:
        tensor: Tensor to hash
    
    Returns:
        Hexadecimal hash string
    """
    # Convert to CPU and contiguous for deterministic hashing
    tensor_cpu = tensor.detach().cpu().contiguous()
    
    # Use numpy bytes for hashing
    tensor_bytes = tensor_cpu.numpy().tobytes()
    
    # Hash shape and dtype together with values
    shape_str = str(tuple(tensor_cpu.shape))
    dtype_str = str(tensor_cpu.dtype)
    
    hasher = hashlib.sha256()
    hasher.update(shape_str.encode())
    hasher.update(dtype_str.encode())
    hasher.update(tensor_bytes)
    
    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity


def hash_hyperrectangle(region: HyperRectangle) -> str:
    """
    Compute a hash of a HyperRectangle region.
    
    Args:
        region: HyperRectangle to hash
    
    Returns:
        Hash string uniquely identifying this region
    """
    lower_hash = hash_tensor(region.lower)
    upper_hash = hash_tensor(region.upper)
    
    hasher = hashlib.sha256()
    hasher.update(b"hyperrectangle")
    hasher.update(lower_hash.encode())
    hasher.update(upper_hash.encode())
    
    return hasher.hexdigest()[:16]


def hash_multi_input_region(region: MultiInputRegion) -> str:
    """
    Compute a hash of a MultiInputRegion.
    
    Args:
        region: MultiInputRegion to hash
    
    Returns:
        Hash string uniquely identifying this region
    """
    hasher = hashlib.sha256()
    hasher.update(b"multi_input")
    
    # Sort by input ID for deterministic ordering
    for input_id in sorted(region.keys()):
        input_region = region[input_id]
        hasher.update(str(input_id).encode())
        hasher.update(hash_hyperrectangle(input_region).encode())
    
    return hasher.hexdigest()[:16]


def hash_region(region: AbstractRegion) -> str:
    """
    Compute a hash of any AbstractRegion.
    
    Dispatches to the appropriate hashing function based on region type.
    
    Args:
        region: Region to hash
    
    Returns:
        Hash string uniquely identifying this region
    
    Raises:
        TypeError: If region type is not supported
    """
    from bound_propagation.regions.hyperrectangle import HyperRectangle
    from bound_propagation.regions.multi_input import MultiInputRegion
    
    if isinstance(region, MultiInputRegion):
        return hash_multi_input_region(region)
    elif isinstance(region, HyperRectangle):
        return hash_hyperrectangle(region)
    else:
        raise TypeError(f"Unsupported region type for hashing: {type(region)}")


def regions_equal(region1: AbstractRegion, region2: AbstractRegion) -> bool:
    """
    Check if two regions are equal.
    
    Args:
        region1: First region
        region2: Second region
    
    Returns:
        True if regions are equal
    """
    from bound_propagation.regions.hyperrectangle import HyperRectangle
    from bound_propagation.regions.multi_input import MultiInputRegion
    
    # Different types are not equal
    if type(region1) != type(region2):
        return False
    
    if isinstance(region1, HyperRectangle):
        assert isinstance(region2, HyperRectangle)
        return (
            torch.allclose(region1.lower, region2.lower, rtol=1e-6, atol=1e-8)
            and torch.allclose(region1.upper, region2.upper, rtol=1e-6, atol=1e-8)
        )
    elif isinstance(region1, MultiInputRegion):
        assert isinstance(region2, MultiInputRegion)
        
        # Must have same input IDs
        if set(region1.keys()) != set(region2.keys()):
            return False
        
        # Check each input region
        for input_id in region1.keys():
            if not regions_equal(region1[input_id], region2[input_id]):
                return False
        
        return True
    else:
        # Fall back to hash comparison
        return hash_region(region1) == hash_region(region2)


def region_contains(outer: HyperRectangle, inner: HyperRectangle) -> bool:
    """
    Check if outer region contains inner region.
    
    This is used for conservative cache reuse: if we have cached bounds
    for a larger region, they are valid (though possibly loose) for
    any smaller region contained within it.
    
    Args:
        outer: Potentially larger region
        inner: Potentially smaller region
    
    Returns:
        True if outer contains inner (outer.lower <= inner.lower and inner.upper <= outer.upper)
    """
    # Must have same shape
    if outer.shape != inner.shape:
        return False
    
    # Check containment
    lower_ok = torch.all(outer.lower <= inner.lower + 1e-6)
    upper_ok = torch.all(inner.upper <= outer.upper + 1e-6)
    
    return bool(lower_ok and upper_ok)
