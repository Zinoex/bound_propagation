from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.transpose import ForwardLBPTransposeStrategy
from bound_propagation.regions import HyperRectangle


def test_transpose_2d() -> None:
    """Test transpose on 2D tensor."""
    # Region: 4 elements for shape (2, 2)
    # [[x0, x1],  -> [[x0, x2],
    #  [x2, x3]]      [x1, x3]]
    region = HyperRectangle(
        lower=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        upper=torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )

    # Identity linear bounds for (2, 2)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(4).view(2, 2, 4),
        bias_lower=torch.zeros(2, 2),
        linear_upper=torch.eye(4).view(2, 2, 4),
        bias_upper=torch.zeros(2, 2),
    )

    strategy = ForwardLBPTransposeStrategy()

    class MockNode:
        def __init__(self):
            # transpose(0, 1) for 2D
            self.attributes = {"dim0": 0, "dim1": 1}

    node = MockNode()

    result = strategy.propagate_forwards(node=node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Linear structure should be preserved (just permuted)
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    lower, upper = result.concretize()
    # Original: [[1, 2], [3, 4]] to [[2, 3], [4, 5]]
    # Transposed: [[1, 3], [2, 4]] to [[2, 4], [3, 5]]
    assert lower.shape == (2, 2)
    assert upper.shape == (2, 2)
    expected_lower = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    expected_upper = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    assert torch.allclose(lower, expected_lower)
    assert torch.allclose(upper, expected_upper)


def test_transpose_3d() -> None:
    """Test transpose on 3D tensor."""
    # Region: 8 elements for shape (2, 2, 2)
    region = HyperRectangle(
        lower=torch.arange(0.0, 8.0),
        upper=torch.arange(1.0, 9.0),
    )

    # Identity linear bounds
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(8).view(2, 2, 2, 8),
        bias_lower=torch.zeros(2, 2, 2),
        linear_upper=torch.eye(8).view(2, 2, 2, 8),
        bias_upper=torch.zeros(2, 2, 2),
    )

    strategy = ForwardLBPTransposeStrategy()

    class MockNode:
        def __init__(self):
            # transpose(0, 2)
            self.attributes = {"dim0": 0, "dim1": 2}

    node = MockNode()

    result = strategy.propagate_forwards(node=node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Linear structure preserved
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    lower, upper = result.concretize()
    # Shape should be transposed: (2, 2, 2) with dims 0,2 swapped -> still (2, 2, 2)
    assert lower.shape == (2, 2, 2)
    assert upper.shape == (2, 2, 2)


def test_transpose_identity() -> None:
    """Test transpose with same dimensions (identity)."""
    # Region: 4 elements for shape (2, 2)
    region = HyperRectangle(
        lower=torch.ones(4),
        upper=torch.full((4,), 2.0),
    )

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(4).view(2, 2, 4),
        bias_lower=torch.zeros(2, 2),
        linear_upper=torch.eye(4).view(2, 2, 4),
        bias_upper=torch.zeros(2, 2),
    )

    strategy = ForwardLBPTransposeStrategy()

    class MockNode:
        def __init__(self):
            # transpose(1, 1) - same dimension
            self.attributes = {"dim0": 1, "dim1": 1}

    node = MockNode()

    result = strategy.propagate_forwards(node=node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should be unchanged
    assert torch.allclose(lower, torch.ones(2, 2))
    assert torch.allclose(upper, torch.full((2, 2), 2.0))


def test_transpose_with_bias() -> None:
    """Test transpose with non-identity linear bounds."""
    # Region: x ∈ [0, 1] for 4 elements
    region = HyperRectangle(
        lower=torch.zeros(4),
        upper=torch.ones(4),
    )

    # Non-identity bounds: add bias
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(4).view(2, 2, 4),
        bias_lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        linear_upper=torch.eye(4).view(2, 2, 4),
        bias_upper=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
    )

    strategy = ForwardLBPTransposeStrategy()

    class MockNode:
        def __init__(self):
            self.attributes = {"dim0": 0, "dim1": 1}

    node = MockNode()

    result = strategy.propagate_forwards(node=node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Bias should also be transposed
    assert torch.allclose(result.bias_lower, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([[2.0, 4.0], [3.0, 5.0]]))
