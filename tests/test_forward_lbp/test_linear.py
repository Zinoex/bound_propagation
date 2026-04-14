from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.linear import ForwardLBPLinear
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


def test_linear_simple_transformation() -> None:
    """Test simple linear transformation."""
    # Region: x ∈ [1, 2]
    # Linear: y = 2x + 3
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[2.0]])
    bias = torch.tensor([3.0])

    strategy = ForwardLBPLinear()
    result = propagate(strategy, bounds, weight=weight, bias=bias)

    # Expected: y = 2x + 3, so at x=1: y=5, at x=2: y=7
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([7.0]))


def test_linear_multi_input_multi_output() -> None:
    """Test linear layer with multiple inputs and outputs."""
    # Region: x0 ∈ [0, 1], x1 ∈ [2, 3]
    # Linear: y = [x0 + x1, 2*x0 - x1] + [1, -1]
    region = HyperRectangle(lower=torch.tensor([0.0, 2.0]), upper=torch.tensor([1.0, 3.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0, 1.0], [2.0, -1.0]])
    bias = torch.tensor([1.0, -1.0])

    strategy = ForwardLBPLinear()
    result = propagate(strategy, bounds, weight=weight, bias=bias)

    lower, upper = result.concretize()
    # First output: x0 + x1 + 1
    # At (x0, x1) = (0, 2): 0 + 2 + 1 = 3
    # At (x0, x1) = (1, 3): 1 + 3 + 1 = 5
    # Second output: 2*x0 - x1 - 1
    # At (x0, x1) = (0, 3): 2*0 - 3 - 1 = -4 (but need to check all corners)
    # At (x0, x1) = (1, 2): 2*1 - 2 - 1 = -1
    # At (x0, x1) = (0, 2): 2*0 - 2 - 1 = -3
    # At (x0, x1) = (1, 3): 2*1 - 3 - 1 = -2
    # So second output: [-4, -1]
    assert torch.allclose(lower, torch.tensor([3.0, -4.0]))
    assert torch.allclose(upper, torch.tensor([5.0, -1.0]))


def test_linear_no_bias() -> None:
    """Test linear transformation without bias."""
    # Region: x ∈ [2, 4]
    # Linear: y = 3x (no bias)
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[3.0]])

    strategy = ForwardLBPLinear()
    result = propagate(strategy, bounds, weight=weight, bias=None)

    # Expected: y = 3x, so at x=2: y=6, at x=4: y=12
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([6.0]))
    assert torch.allclose(upper, torch.tensor([12.0]))


def test_linear_negative_weights() -> None:
    """Test linear transformation with negative weights."""
    # Region: x ∈ [1, 3]
    # Linear: y = -2x + 5
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[-2.0]])
    bias = torch.tensor([5.0])

    strategy = ForwardLBPLinear()
    result = propagate(strategy, bounds, weight=weight, bias=bias)

    # Expected: y = -2x + 5
    # At x=1: y = -2 + 5 = 3
    # At x=3: y = -6 + 5 = -1
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-1.0]))
    assert torch.allclose(upper, torch.tensor([3.0]))
