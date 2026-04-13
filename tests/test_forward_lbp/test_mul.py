from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.mul import (
    ForwardLBPMul,
    ForwardLBPMulWithConstant,
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


def test_mul_abstract_abstract_concretizes() -> None:
    """Test element-wise multiplication of two abstract bounds concretizes."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Bounds A: identity for x0
    # Bounds B: identity for x1
    # Multiplication is non-linear, so must concretize
    # Result: interval [1*3, 2*4] = [3, 8]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))

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

    strategy = ForwardLBPMul()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds_a, bounds_b])  # ty:ignore[invalid-argument-type]

    # Should have no linear dependency (concretized)
    assert result.linear_lower is None
    assert result.linear_upper is None

    # First element: [1, 2] * [0, 0] = [0, 0]
    # Second element: [0, 0] * [3, 4] = [0, 0]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([0.0, 0.0]))


def test_mul_abstract_constant_positive_scalar() -> None:
    """Test multiplication by a positive scalar constant."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x + 1, upper = x + 3
    # Constant: 2
    # Result: lower = 2(x + 1) = 2x + 2 ∈ [6, 12]
    #         upper = 2(x + 3) = 2x + 6 ∈ [10, 16]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([3.0]),
    )

    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 2.0])  # ty:ignore[invalid-argument-type]

    # Linear: 1 * 2 = 2
    assert torch.allclose(result.linear_lower, torch.tensor([[2.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0]]))
    # Bias: 1 * 2 = 2, 3 * 2 = 6
    assert torch.allclose(result.bias_lower, torch.tensor([2.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([6.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([6.0]))
    assert torch.allclose(upper, torch.tensor([16.0]))


def test_mul_abstract_constant_negative_scalar() -> None:
    """Test multiplication by a negative scalar constant."""
    # Region: x ∈ [1, 4]
    # Bounds: lower = x, upper = x
    # Constant: -2
    # Result: lower = -2x ∈ [-8, -2], upper = -2x ∈ [-8, -2]
    # Note: negative multiplication swaps lower and upper
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, -2.0])  # ty:ignore[invalid-argument-type]

    # Linear: 1 * -2 = -2 (swapped because negative)
    assert torch.allclose(result.linear_lower, torch.tensor([[-2.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-2.0]]))
    # Bias: 0 * -2 = 0 (swapped)
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    # At x=1: -2*1 = -2, At x=4: -2*4 = -8
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-8.0]))
    assert torch.allclose(upper, torch.tensor([-2.0]))


def test_mul_abstract_constant_zero() -> None:
    """Test multiplication by zero."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x, upper = x
    # Constant: 0
    # Result: lower = 0, upper = 0
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, 0.0])  # ty:ignore[invalid-argument-type]

    # Linear dependency removed (multiplied by 0)
    assert result.linear_lower is None
    assert result.linear_upper is None
    # Bias: 0 * 0 = 0
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([0.0]))


def test_mul_abstract_constant_tensor_positive() -> None:
    """Test multiplication by a positive tensor constant."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Bounds: identity
    # Constant: [2, 3]
    # Result: x * [2, 3] = [2*x0, 3*x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([2.0, 3.0])
    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, constant])  # ty:ignore[invalid-argument-type]

    # Linear: [1, 0] * 2 = [2, 0], [0, 1] * 3 = [0, 3]
    assert torch.allclose(result.linear_lower, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    # Bias: 0 * [2, 3] = [0, 0]
    assert torch.allclose(result.bias_lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0, 0.0]))

    # [1, 3] * [2, 3] = [2, 9], [2, 4] * [2, 3] = [4, 12]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0, 9.0]))
    assert torch.allclose(upper, torch.tensor([4.0, 12.0]))


def test_mul_abstract_constant_tensor_mixed_signs() -> None:
    """Test multiplication by a tensor with mixed signs."""
    # Region: x0 ∈ [1, 3], x1 ∈ [2, 4]
    # Bounds: identity
    # Constant: [2, -1]
    # Result: [2*x0, -x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([2.0, -1.0])
    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds, constant])  # ty:ignore[invalid-argument-type]

    # Linear: [1, 0] * 2 = [2, 0], [0, 1] * -1 = [0, -1]
    # For x1: negative multiplier swaps bounds
    assert torch.allclose(result.linear_lower, torch.tensor([[2.0, 0.0], [0.0, -1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0, 0.0], [0.0, -1.0]]))

    # [1, 2] * [2, -1] = [2, -2], [3, 4] * [2, -1] = [6, -4]
    # lower = [2, -4], upper = [6, -2]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0, -4.0]))
    assert torch.allclose(upper, torch.tensor([6.0, -2.0]))


def test_mul_constant_abstract() -> None:
    """Test multiplication with constant as first operand (commutative)."""
    # Region: x ∈ [2, 5]
    # Constant: 3
    # Bounds: lower = x, upper = x
    # Result: 3 * x = 3x ∈ [6, 15]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMulWithConstant()
    result = strategy.propagate_forwards(node=None, input_bounds=[3.0, bounds])  # ty:ignore[invalid-argument-type]

    # Linear: 3 * 1 = 3
    assert torch.allclose(result.linear_lower, torch.tensor([[3.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[3.0]]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([6.0]))
    assert torch.allclose(upper, torch.tensor([15.0]))
