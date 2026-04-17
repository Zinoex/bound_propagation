from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.linear import (
    ForwardLBPSub,
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


def test_sub_abstract_abstract() -> None:
    """Test subtraction of two linear bounds."""
    # Region: x0 ∈ [1, 3], x1 ∈ [2, 4]
    # Bounds A: lower = 2*x0, upper = 2*x0
    # Bounds B: lower = x1, upper = x1
    # Result: lower = 2*x0 - x1 ∈ [2*1 - 4, 2*3 - 2] = [-2, 4]
    #         upper = 2*x0 - x1 ∈ [2*1 - 4, 2*3 - 2] = [-2, 4]
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[2.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[2.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPSub()
    result = propagate(strategy, bounds_a, bounds_b)

    # Linear: [2, 0] - [0, 1] = [2, -1]
    assert torch.allclose(result.linear_lower, torch.tensor([[2.0, -1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0, -1.0]]))
    # Bias: 0 - 0 = 0
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    # At lower corner [1, 2]: 2*1 - 1*2 = 0
    # At upper corner [3, 4]: 2*3 - 1*4 = 2
    # At [1, 4]: 2*1 - 1*4 = -2 (minimum)
    # At [3, 2]: 2*3 - 1*2 = 4 (maximum)
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-2.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))


def test_sub_abstract_constant_scalar() -> None:
    """Test subtraction of a scalar constant from linear bounds."""
    # Region: x ∈ [5, 8]
    # Bounds: lower = x + 1, upper = x + 2
    # Constant: 3
    # Result: lower = (x + 1) - 3 = x - 2 ∈ [3, 6]
    #         upper = (x + 2) - 3 = x - 1 ∈ [4, 7]
    region = HyperRectangle(lower=torch.tensor([5.0]), upper=torch.tensor([8.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([2.0]),
    )

    strategy = ForwardLBPSub()
    result = propagate(strategy, bounds, 3.0)

    # Linear unchanged
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    # Bias: 1 - 3 = -2, 2 - 3 = -1
    assert torch.allclose(result.bias_lower, torch.tensor([-2.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([-1.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([7.0]))


def test_sub_constant_abstract_scalar() -> None:
    """Test subtraction where constant is the left operand."""
    # Region: x ∈ [2, 5]
    # Constant: 10
    # Bounds: lower = x, upper = x
    # Result: lower = 10 - x ∈ [10 - 5, 10 - 2] = [5, 8]
    #         upper = 10 - x ∈ [10 - 5, 10 - 2] = [5, 8]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSub()
    result = propagate(strategy, 10.0, bounds)

    # Linear: 0 - [1] = -1 (negated)
    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-1.0]]))
    # Bias: 10 - 0 = 10
    assert torch.allclose(result.bias_lower, torch.tensor([10.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([10.0]))

    # At x=2: 10 - 2 = 8, At x=5: 10 - 5 = 5
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([8.0]))


def test_sub_abstract_constant_tensor() -> None:
    """Test subtraction of a tensor constant."""
    # Region: x0 ∈ [1, 3], x1 ∈ [2, 4]
    # Bounds: identity
    # Constant: [1, 2]
    # Result: x - [1, 2] = [x0-1, x1-2]
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([1.0, 2.0])
    strategy = ForwardLBPSub()
    result = propagate(strategy, bounds, constant)

    # Linear unchanged (identity)
    assert torch.allclose(result.linear_lower, torch.eye(2))
    assert torch.allclose(result.linear_upper, torch.eye(2))
    # Bias: [0, 0] - [1, 2] = [-1, -2]
    assert torch.allclose(result.bias_lower, torch.tensor([-1.0, -2.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([-1.0, -2.0]))

    # [1, 2] - [1, 2] = [0, 0], [3, 4] - [1, 2] = [2, 2]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 2.0]))


def test_sub_constant_abstract_tensor() -> None:
    """Test subtraction with tensor constant as left operand."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Constant: [10, 20]
    # Bounds: identity
    # Result: [10, 20] - [x0, x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([10.0, 20.0])
    strategy = ForwardLBPSub()
    result = propagate(strategy, constant, bounds)

    # Linear: negated identity
    assert torch.allclose(result.linear_lower, -torch.eye(2))
    assert torch.allclose(result.linear_upper, -torch.eye(2))
    # Bias: [10, 20] - [0, 0] = [10, 20]
    assert torch.allclose(result.bias_lower, torch.tensor([10.0, 20.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([10.0, 20.0]))

    # [10, 20] - [2, 4] = [8, 16] (lower), [10, 20] - [1, 3] = [9, 17] (upper)
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([8.0, 16.0]))
    assert torch.allclose(upper, torch.tensor([9.0, 17.0]))


def test_sub_with_bias() -> None:
    """Test subtraction with non-zero bias terms."""
    # Region: x ∈ [0, 2]
    # Bounds A: lower = 3x + 2, upper = 3x + 5
    # Bounds B: lower = x + 1, upper = x + 2
    # Result: lower = (3x + 2) - (x + 2) = 2x + 0 ∈ [0, 4]
    #         upper = (3x + 5) - (x + 1) = 2x + 4 ∈ [4, 8]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[3.0]]),
        bias_lower=torch.tensor([2.0]),
        linear_upper=torch.tensor([[3.0]]),
        bias_upper=torch.tensor([5.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([2.0]),
    )

    strategy = ForwardLBPSub()
    result = propagate(strategy, bounds_a, bounds_b)

    # Linear: 3 - 1 = 2
    assert torch.allclose(result.linear_lower, torch.tensor([[2.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0]]))
    # Bias: 2 - 2 = 0, 5 - 1 = 4
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([4.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([8.0]))
