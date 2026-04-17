from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.shape import ForwardLBPFlatten
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def test_flatten_2d_to_1d() -> None:
    """Test flattening a 2D tensor to 1D."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (2, 3)
    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(2, 3, 1),
        bias_lower=torch.arange(6.0).view(2, 3),
        linear_upper=torch.ones(2, 3, 1),
        bias_upper=torch.arange(6.0).view(2, 3) + 1,
    )

    strategy = ForwardLBPFlatten()
    result = propagate(strategy, bounds)

    # Shape should be flattened to (6,)
    assert result.bias_lower.shape == (6,)
    assert result.bias_upper.shape == (6,)
    assert torch.allclose(result.bias_lower, torch.arange(6.0))
    assert torch.allclose(result.bias_upper, torch.arange(6.0) + 1)


def test_flatten_3d_to_1d() -> None:
    """Test flattening a 3D tensor to 1D."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (2, 2, 2)
    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(2, 2, 2, 1),
        bias_lower=torch.arange(8.0).view(2, 2, 2),
        linear_upper=torch.ones(2, 2, 2, 1),
        bias_upper=torch.arange(8.0).view(2, 2, 2) + 2,
    )

    strategy = ForwardLBPFlatten()
    result = propagate(strategy, bounds)

    # Shape should be flattened to (8,)
    assert result.bias_lower.shape == (8,)
    assert result.bias_upper.shape == (8,)
    assert torch.allclose(result.bias_lower, torch.arange(8.0))
    assert torch.allclose(result.bias_upper, torch.arange(8.0) + 2)


def test_flatten_1d_identity() -> None:
    """Test flattening a 1D tensor (should be identity)."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (5,)
    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.arange(5.0),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.arange(5.0) + 1,
    )

    strategy = ForwardLBPFlatten()
    result = propagate(strategy, bounds)

    # Shape should remain (5,)
    assert result.bias_lower.shape == (5,)
    assert result.bias_upper.shape == (5,)
    assert torch.allclose(result.bias_lower, torch.arange(5.0))
    assert torch.allclose(result.bias_upper, torch.arange(5.0) + 1)
