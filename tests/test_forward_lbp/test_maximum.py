from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.maximum import (
    ForwardLBPMaximumStrategy,
    ForwardLBPMaximumWithConstant,
)
from bound_propagation.regions import HyperRectangle


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        region=region,
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_maximum_abstract_abstract_concretizes() -> None:
    """Test element-wise maximum of two abstract bounds concretizes."""
    # Region: x0 ∈ [1, 3], x1 ∈ [2, 4]
    # Bounds A: [x0, x0]
    # Bounds B: [x1, x1]
    # max(x0, x1): Since x0 and x1 vary independently, we must concretize
    # Result: [max(1, 2), max(3, 4)] = [2, 4]
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))

    bounds_a = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )
    bounds_b = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )

    strategy = ForwardLBPMaximumStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds_a, bounds_b])  # ty:ignore[invalid-argument-type]

    # Should concretize (lose linear dependency)
    assert result.linear_lower is None
    assert result.linear_upper is None

    # First element: max([1, 3], [0, 0]) = [1, 3]
    # Second element: max([0, 0], [2, 4]) = [2, 4]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0, 2.0]))
    assert torch.allclose(upper, torch.tensor([3.0, 4.0]))


def test_maximum_abstract_constant_positive() -> None:
    """Test element-wise maximum with a positive constant."""
    # Region: x ∈ [0, 5]
    # Bounds: lower = x, upper = x
    # Constant: 2
    # max(x, 2): [max(0, 2), max(5, 2)] = [2, 5]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 2.0])  # ty:ignore[invalid-argument-type]

    # Concretizes
    assert result.linear_lower is None
    assert result.linear_upper is None

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_maximum_abstract_constant_below_range() -> None:
    """Test maximum with constant below the input range."""
    # Region: x ∈ [3, 7]
    # Bounds: lower = x, upper = x
    # Constant: 1
    # max(x, 1) = x (since x >= 3 > 1)
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([7.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 1.0])  # ty:ignore[invalid-argument-type]

    # Still concretizes but result is the input interval
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([7.0]))


def test_maximum_abstract_constant_above_range() -> None:
    """Test maximum with constant above the input range."""
    # Region: x ∈ [1, 3]
    # Bounds: lower = x, upper = x
    # Constant: 5
    # max(x, 5) = 5 (since x <= 3 < 5)
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 5.0])  # ty:ignore[invalid-argument-type]

    # Result is constant 5
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_maximum_abstract_constant_tensor() -> None:
    """Test maximum with a tensor constant."""
    # Region: x0 ∈ [0, 4], x1 ∈ [1, 5]
    # Bounds: identity
    # Constant: [2, 3]
    # max([x0, x1], [2, 3]): ([max(0, 2), max(4, 2)], [max(1, 3), max(5, 3)]) = ([2, 4], [3, 5])
    region = HyperRectangle(lower=torch.tensor([0.0, 1.0]), upper=torch.tensor([4.0, 5.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([2.0, 3.0])
    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, constant])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0, 3.0]))
    assert torch.allclose(upper, torch.tensor([4.0, 5.0]))


def test_maximum_constant_abstract() -> None:
    """Test maximum with constant as first operand (commutative)."""
    # Region: x ∈ [1, 6]
    # Constant: 3
    # Bounds: lower = x, upper = x
    # max(3, x): [max(3, 1), max(3, 6)] = [3, 6]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([6.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[3.0, bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([6.0]))


def test_maximum_negative_values() -> None:
    """Test maximum with negative values."""
    # Region: x ∈ [-5, -1]
    # Bounds: lower = x, upper = x
    # Constant: -3
    # max(x, -3): [max(-5, -3), max(-1, -3)] = [-3, -1]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, -3.0])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-3.0]))
    assert torch.allclose(upper, torch.tensor([-1.0]))
