from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.squeeze import ForwardLBPSqueeze
from bound_propagation.regions import HyperRectangle


def test_squeeze_single_dim() -> None:
    """Test squeeze on a specific dimension."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (1, 3, 1, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(1, 3, 1, 4, 1),
        bias_lower=torch.arange(12.0).view(1, 3, 1, 4),
        linear_upper=torch.ones(1, 3, 1, 4, 1),
        bias_upper=(torch.arange(12.0) + 1).view(1, 3, 1, 4),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 0}

    node = MockNode()
    strategy = ForwardLBPSqueeze()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should squeeze dim 0: (1, 3, 1, 4) -> (3, 1, 4)
    assert result.bias_lower.shape == (3, 1, 4)
    assert result.bias_upper.shape == (3, 1, 4)


def test_squeeze_all_dims() -> None:
    """Test squeeze without specifying dimension (removes all size-1 dims)."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (1, 3, 1, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(1, 3, 1, 4, 1),
        bias_lower=torch.arange(12.0).view(1, 3, 1, 4),
        linear_upper=torch.ones(1, 3, 1, 4, 1),
        bias_upper=(torch.arange(12.0) + 1).view(1, 3, 1, 4),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": None}

    node = MockNode()
    strategy = ForwardLBPSqueeze()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should squeeze all singleton dims: (1, 3, 1, 4) -> (3, 4)
    assert result.bias_lower.shape == (3, 4)
    assert result.bias_upper.shape == (3, 4)


def test_squeeze_middle_dim() -> None:
    """Test squeeze on a middle dimension."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (2, 1, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 1, 3, 1),
        bias_lower=torch.arange(6.0).view(2, 1, 3),
        linear_upper=torch.ones(2, 1, 3, 1),
        bias_upper=(torch.arange(6.0) + 1).view(2, 1, 3),
    )

