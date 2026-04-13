from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.select import ForwardLBPSelect
from bound_propagation.regions import HyperRectangle


def test_select_dim0() -> None:
    """Test select along dimension 0."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (3, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(3, 4, 1),
        bias_lower=torch.arange(12.0).view(3, 4),
        linear_upper=torch.ones(3, 4, 1),
        bias_upper=(torch.arange(12.0) + 1).view(3, 4),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 0, "index": 1}

    node = MockNode()
    strategy = ForwardLBPSelect()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should select row 1
    lower, upper = result.concretize()
    assert lower.shape == (4,)
    assert upper.shape == (4,)
    assert torch.allclose(lower, torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.allclose(upper, torch.tensor([5.0, 6.0, 7.0, 8.0]))


def test_select_dim1() -> None:
    """Test select along dimension 1."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (3, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(3, 4, 1),
        bias_lower=torch.arange(12.0).view(3, 4),
        linear_upper=torch.ones(3, 4, 1),
        bias_upper=(torch.arange(12.0) + 1).view(3, 4),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 1, "index": 2}

    node = MockNode()
    strategy = ForwardLBPSelect()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should select column 2 from each row
    lower, upper = result.concretize()
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert torch.allclose(lower, torch.tensor([2.0, 6.0, 10.0]))
    assert torch.allclose(upper, torch.tensor([3.0, 7.0, 11.0]))


def test_select_3d() -> None:
    """Test select on 3D tensor."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (2, 3, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 3, 4, 1),
        bias_lower=torch.arange(24.0).view(2, 3, 4),
        linear_upper=torch.ones(2, 3, 4, 1),
        bias_upper=(torch.arange(24.0) + 1).view(2, 3, 4),
    )

    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 1, "index": 1}

    node = MockNode()
    strategy = ForwardLBPSelect()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should select index 1 along dim 1
    lower, upper = result.concretize()
    assert lower.shape == (2, 4)
    assert upper.shape == (2, 4)
    # Elements at positions [:,1,:]
    expected_lower = torch.tensor([[4.0, 5.0, 6.0, 7.0], [16.0, 17.0, 18.0, 19.0]])
    expected_upper = torch.tensor([[5.0, 6.0, 7.0, 8.0], [17.0, 18.0, 19.0, 20.0]])
    assert torch.allclose(lower, expected_lower)
    assert torch.allclose(upper, expected_upper)
