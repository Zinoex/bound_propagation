from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.reshape import ForwardLBPReshape
from bound_propagation.regions import HyperRectangle


def test_reshape_1d_to_2d() -> None:
    """Test reshaping 1D tensor to 2D."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (6,)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(6, 1),
        bias_lower=torch.arange(6.0),
        linear_upper=torch.ones(6, 1),
        bias_upper=torch.arange(6.0) + 1,
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"shape": (2, 3)}

    node = MockNode()
    strategy = ForwardLBPReshape()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Shape should be (2, 3)
    assert result.bias_lower.shape == (2, 3)
    assert result.bias_upper.shape == (2, 3)
    assert torch.allclose(result.bias_lower, torch.arange(6.0).view(2, 3))
    assert torch.allclose(result.bias_upper, (torch.arange(6.0) + 1).view(2, 3))


def test_reshape_2d_to_1d() -> None:
    """Test reshaping 2D tensor to 1D."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (2, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 3, 1),
        bias_lower=torch.arange(6.0).view(2, 3),
        linear_upper=torch.ones(2, 3, 1),
        bias_upper=(torch.arange(6.0) + 1).view(2, 3),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"shape": (6,)}

    node = MockNode()
    strategy = ForwardLBPReshape()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Shape should be (6,)
    assert result.bias_lower.shape == (6,)
    assert result.bias_upper.shape == (6,)
    assert torch.allclose(result.bias_lower, torch.arange(6.0))
    assert torch.allclose(result.bias_upper, torch.arange(6.0) + 1)


def test_reshape_3d() -> None:
    """Test reshaping between 3D shapes."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Input shape: (2, 2, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 2, 3, 1),
        bias_lower=torch.arange(12.0).view(2, 2, 3),
        linear_upper=torch.ones(2, 2, 3, 1),
        bias_upper=(torch.arange(12.0) + 1).view(2, 2, 3),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"shape": (3, 4)}

    node = MockNode()
    strategy = ForwardLBPReshape()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Shape should be (3, 4)
    assert result.bias_lower.shape == (3, 4)
    assert result.bias_upper.shape == (3, 4)
    assert torch.allclose(result.bias_lower, torch.arange(12.0).view(3, 4))
    assert torch.allclose(result.bias_upper, (torch.arange(12.0) + 1).view(3, 4))
