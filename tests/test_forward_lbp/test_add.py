from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.linear import (
    ForwardLBPAdd,
)
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        regions=[region],
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_add_abstract_abstract_positive() -> None:
    """Test addition of two linear bounds (both abstract)."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Bounds A: lower = x0, upper = x0 (identity for first variable)
    # Bounds B: lower = x1, upper = x1 (identity for second variable)
    # Result: lower = x0 + x1 ∈ [4, 6], upper = x0 + x1 ∈ [4, 6]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds_a, bounds_b)

    # Linear coefficients: [1, 0] + [0, 1] = [1, 1]
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0, 1.0]]))
    # Biases: 0 + 0 = 0
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    # Concretized bounds: [1, 1] @ [1, 3] + 0 = 4, [1, 1] @ [2, 4] + 0 = 6
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([4.0]))
    assert torch.allclose(upper, torch.tensor([6.0]))


def test_add_abstract_abstract_with_bias() -> None:
    """Test addition with non-zero bias terms."""
    # Region: x ∈ [0, 1]
    # Bounds A: lower = 2x + 1, upper = 2x + 3
    # Bounds B: lower = x + 0.5, upper = x + 1.5
    # Result: lower = 3x + 1.5, upper = 3x + 4.5
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[2.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[2.0]]),
        bias_upper=torch.tensor([3.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([0.5]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([1.5]),
    )

    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds_a, bounds_b)

    # Linear: 2 + 1 = 3
    assert torch.allclose(result.linear_lower, torch.tensor([[3.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[3.0]]))
    # Bias: 1 + 0.5 = 1.5, 3 + 1.5 = 4.5
    assert torch.allclose(result.bias_lower, torch.tensor([1.5]))
    assert torch.allclose(result.bias_upper, torch.tensor([4.5]))

    # At x=0: lower = 3*0 + 1.5 = 1.5, upper = 3*0 + 4.5 = 4.5
    # At x=1: lower = 3*1 + 1.5 = 4.5, upper = 3*1 + 4.5 = 7.5
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.5]))
    assert torch.allclose(upper, torch.tensor([7.5]))


def test_add_abstract_abstract_different_regions() -> None:
    """Test addition merges affine terms from different input regions."""
    region_x = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
    region_y = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([3.0]))

    bounds_x = LinearBounds(
        regions=[region_x],
        linear_lower=[torch.tensor([[1.0]])],
        bias_lower=torch.tensor([0.0]),
        linear_upper=[torch.tensor([[1.0]])],
        bias_upper=torch.tensor([0.0]),
        input_ids=[101],
    )
    bounds_y = LinearBounds(
        regions=[region_y],
        linear_lower=[torch.tensor([[1.0]])],
        bias_lower=torch.tensor([0.5]),
        linear_upper=[torch.tensor([[1.0]])],
        bias_upper=torch.tensor([0.5]),
        input_ids=[202],
    )

    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds_x, bounds_y)

    assert len(result.regions) == 2
    assert result.input_ids == [101, 202]
    assert torch.allclose(result.linear_lowers[0], torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_lowers[1], torch.tensor([[1.0]]))

    lower, upper = result.concretize()
    # x + y + 0.5 over x in [0, 1], y in [2, 3]
    assert torch.allclose(lower, torch.tensor([2.5]))
    assert torch.allclose(upper, torch.tensor([4.5]))


def test_add_abstract_constant_scalar() -> None:
    """Test addition of linear bounds with a scalar constant."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x, upper = x
    # Constant: 3
    # Result: lower = x + 3 ∈ [5, 8], upper = x + 3 ∈ [5, 8]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds, 3.0)

    # Linear coefficients unchanged
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    # Bias increased by 3
    assert torch.allclose(result.bias_lower, torch.tensor([3.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([3.0]))

    # Concretized: [2, 5] + 3 = [5, 8]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([8.0]))


def test_add_abstract_constant_tensor() -> None:
    """Test addition of linear bounds with a tensor constant."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Bounds: identity (lower = x, upper = x)
    # Constant: [10, 20]
    # Result: lower = x + [10, 20], upper = x + [10, 20]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([10.0, 20.0])
    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds, constant)

    # Linear coefficients unchanged (identity)
    assert torch.allclose(result.linear_lower, torch.eye(2))
    assert torch.allclose(result.linear_upper, torch.eye(2))
    # Bias increased by [10, 20]
    assert torch.allclose(result.bias_lower, torch.tensor([10.0, 20.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([10.0, 20.0]))

    # Concretized: [1, 3] + [10, 20] = [11, 23], [2, 4] + [10, 20] = [12, 24]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([11.0, 23.0]))
    assert torch.allclose(upper, torch.tensor([12.0, 24.0]))


def test_add_constant_abstract() -> None:
    """Test addition with constant as first operand (commutative)."""
    # Region: x ∈ [1, 3]
    # Constant: 5
    # Bounds: lower = x, upper = x
    # Result: lower = 5 + x ∈ [6, 8], upper = 5 + x ∈ [6, 8]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAdd()
    result = propagate(strategy, 5.0, bounds)

    # Linear coefficients unchanged
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    # Bias increased by 5
    assert torch.allclose(result.bias_lower, torch.tensor([5.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([5.0]))

    # Concretized: [1, 3] + 5 = [6, 8]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([6.0]))
    assert torch.allclose(upper, torch.tensor([8.0]))


def test_add_negative_constant() -> None:
    """Test addition with a negative constant."""
    # Region: x ∈ [5, 10]
    # Bounds: lower = x, upper = x
    # Constant: -3
    # Result: lower = x - 3 ∈ [2, 7], upper = x - 3 ∈ [2, 7]
    region = HyperRectangle(lower=torch.tensor([5.0]), upper=torch.tensor([10.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAdd()
    result = propagate(strategy, bounds, -3.0)

    # Bias decreased by 3
    assert torch.allclose(result.bias_lower, torch.tensor([-3.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([-3.0]))

    # Concretized: [5, 10] - 3 = [2, 7]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([7.0]))
