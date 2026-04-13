from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.max import ForwardLBPMax
from bound_propagation.regions import HyperRectangle


def test_max_1d_tensor() -> None:
    """Test max reduction on 1D tensor."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": None, "keepdim": False}

    node = MockNode()
    strategy = ForwardLBPMax()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Max of [1, 2, 3, 4, 5] to [2, 3, 4, 5, 6] is [5, 6]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor(5.0))
    assert torch.allclose(upper, torch.tensor(6.0))


def test_max_along_dim() -> None:
    """Test max reduction along a specific dimension."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (2, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 3, 1),
        bias_lower=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        linear_upper=torch.ones(2, 3, 1),
        bias_upper=torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 1, "keepdim": False}

    node = MockNode()
    strategy = ForwardLBPMax()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Max along dim=1: [3, 4] to [4, 7]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0, 6.0]))
    assert torch.allclose(upper, torch.tensor([4.0, 7.0]))


def test_max_keepdim() -> None:
    """Test max reduction with keepdim=True."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(3, 1),
        bias_lower=torch.tensor([1.0, 2.0, 3.0]),
        linear_upper=torch.ones(3, 1),
        bias_upper=torch.tensor([2.0, 3.0, 4.0]),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 0, "keepdim": True}

    node = MockNode()
    strategy = ForwardLBPMax()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert lower.shape == (1,)
    assert upper.shape == (1,)
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))
