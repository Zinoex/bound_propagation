from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.cat import ForwardLBPConcat
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle, shape: tuple[int, ...]) -> LinearBounds:
    """Create identity linear bounds from a region with specific shape."""
    dim = region.lower.numel()
    return LinearBounds(
        region=region,
        linear_lower=torch.eye(dim).view(shape + (dim,)),
        bias_lower=torch.zeros(shape),
        linear_upper=torch.eye(dim).view(shape + (dim,)),
        bias_upper=torch.zeros(shape),
    )


def test_cat_two_tensors_dim0() -> None:
    """Test concatenating two tensors along dimension 0."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Concat two 1D tensors along dim 0
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))

    # First tensor: x0
    bounds1 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    # Second tensor: x1
    bounds2 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPConcat()
    result = propagate(strategy, [bounds1, bounds2], dim=0)

    # Concretized result should be [x0, x1] = [1, 3] to [2, 4]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0, 3.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 4.0]))


def test_cat_two_tensors_dim1() -> None:
    """Test concatenating two tensors along dimension 1."""
    # Region: x0 ∈ [0, 1], x1 ∈ [2, 3]
    region = HyperRectangle(lower=torch.tensor([0.0, 2.0]), upper=torch.tensor([1.0, 3.0]))

    # First tensor: [[x0], [x1]]
    bounds1 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        bias_lower=torch.tensor([[0.0], [0.0]]),
        linear_upper=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        bias_upper=torch.tensor([[0.0], [0.0]]),
    )

    # Second tensor: [[x0], [x1]]
    bounds2 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        bias_lower=torch.tensor([[0.0], [0.0]]),
        linear_upper=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        bias_upper=torch.tensor([[0.0], [0.0]]),
    )

    strategy = ForwardLBPConcat()
    result = propagate(strategy, [bounds1, bounds2], dim=1)

    # Result should be [[x0, x0], [x1, x1]]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    assert torch.allclose(upper, torch.tensor([[1.0, 1.0], [3.0, 3.0]]))


def test_cat_three_tensors() -> None:
    """Test concatenating three tensors."""
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))

    bounds1 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    bounds2 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([2.0]),
    )

    bounds3 = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([2.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([3.0]),
    )

    strategy = ForwardLBPConcat()
    result = propagate(strategy, [bounds1, bounds2, bounds3], dim=0)

    # Expected: [x, x+1, x+2] to [x, x+2, x+3]
    # At x=1: [1, 2, 3] to [1, 3, 4]
    # At x=2: [2, 3, 4] to [2, 4, 5]
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 4.0, 5.0]))


def test_cat_single_tensor() -> None:
    """Test concatenating a single tensor (edge case)."""
    region = HyperRectangle(lower=torch.tensor([5.0]), upper=torch.tensor([10.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPConcat()
    result = propagate(strategy, [bounds], dim=0)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([10.0]))
