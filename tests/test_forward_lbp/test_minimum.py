from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.minimum import (
    ForwardLBPMinimumStrategy,
    ForwardLBPMinimumWithConstant,
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


def test_minimum_abstract_abstract_concretizes() -> None:
    """Test element-wise minimum of two abstract bounds concretizes."""
    # Region: x0 ∈ [1, 3], x1 ∈ [2, 4]
    # Bounds A: [x0, x0]
    # Bounds B: [x1, x1]
    # min(x0, x1): Must concretize due to independent variation
    # Result: [min(1, 2), min(3, 4)] = [1, 3]
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

    strategy = ForwardLBPMinimumStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds_a, bounds_b])  # ty:ignore[invalid-argument-type]

    # Should concretize
    assert result.linear_lower is None
    assert result.linear_upper is None

    # First element: min([1, 3], [0, 0]) = [0, 0]
    # Second element: min([0, 0], [2, 4]) = [0, 0]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([0.0, 0.0]))


def test_minimum_abstract_constant_positive() -> None:
    """Test element-wise minimum with a positive constant."""
    # Region: x ∈ [0, 5]
    # Bounds: lower = x, upper = x
    # Constant: 3
    # min(x, 3): [min(0, 3), min(5, 3)] = [0, 3]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 3.0])  # ty:ignore[invalid-argument-type]

    # Concretizes
    assert result.linear_lower is None
    assert result.linear_upper is None

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([3.0]))


def test_minimum_abstract_constant_above_range() -> None:
    """Test minimum with constant above the input range."""
    # Region: x ∈ [1, 3]
    # Bounds: lower = x, upper = x
    # Constant: 5
    # min(x, 5) = x (since x <= 3 < 5)
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 5.0])  # ty:ignore[invalid-argument-type]

    # Result is the input interval
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]))
    assert torch.allclose(upper, torch.tensor([3.0]))


def test_minimum_abstract_constant_below_range() -> None:
    """Test minimum with constant below the input range."""
    # Region: x ∈ [3, 7]
    # Bounds: lower = x, upper = x
    # Constant: 1
    # min(x, 1) = 1 (since x >= 3 > 1)
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([7.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 1.0])  # ty:ignore[invalid-argument-type]

    # Result is constant 1
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]))
    assert torch.allclose(upper, torch.tensor([1.0]))


def test_minimum_abstract_constant_tensor() -> None:
    """Test minimum with a tensor constant."""
    # Region: x0 ∈ [0, 4], x1 ∈ [1, 5]
    # Bounds: identity
    # Constant: [2, 3]
    # min([x0, x1], [2, 3]): ([min(0, 2), min(4, 2)], [min(1, 3), min(5, 3)]) = ([0, 2], [1, 3])
    region = HyperRectangle(lower=torch.tensor([0.0, 1.0]), upper=torch.tensor([4.0, 5.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([2.0, 3.0])
    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, constant])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0, 1.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 3.0]))


def test_minimum_constant_abstract() -> None:
    """Test minimum with constant as first operand (commutative)."""
    # Region: x ∈ [1, 6]
    # Constant: 4
    # Bounds: lower = x, upper = x
    # min(4, x): [min(4, 1), min(4, 6)] = [1, 4]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([6.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[4.0, bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))


def test_minimum_negative_values() -> None:
    """Test minimum with negative values."""
    # Region: x ∈ [-5, -1]
    # Bounds: lower = x, upper = x
    # Constant: -3
    # min(x, -3): [min(-5, -3), min(-1, -3)] = [-5, -3]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, -3.0])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-5.0]))
    assert torch.allclose(upper, torch.tensor([-3.0]))


def test_minimum_crossing_zero() -> None:
    """Test minimum with interval crossing zero."""
    # Region: x ∈ [-2, 3]
    # Bounds: lower = x, upper = x
    # Constant: 0
    # min(x, 0): [min(-2, 0), min(3, 0)] = [-2, 0]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMinimumWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 0.0])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-2.0]))
    assert torch.allclose(upper, torch.tensor([0.0]))
