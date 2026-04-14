from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.matmul import (
    ForwardLBPMatmul,
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


def test_matmul_abstract_times_constant() -> None:
    """Test matmul: abstract @ constant."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Matmul: [x0, x1] @ [[1], [2]] = [x0 + 2*x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0], [2.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds, weight)

    # Expected: x0 + 2*x1
    # At (x0, x1) = (1, 3): 1 + 6 = 7
    # At (x0, x1) = (2, 4): 2 + 8 = 10
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([7.0]))
    assert torch.allclose(upper, torch.tensor([10.0]))


def test_matmul_constant_times_abstract() -> None:
    """Test matmul: constant @ abstract."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Matmul: [[1, 2], [3, 1]] @ [x0, x1] = [x0 + 2*x1, 3*x0 + x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0, 2.0], [3.0, 1.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, weight, bounds)

    lower, upper = result.concretize()
    # First output: x0 + 2*x1
    # At (x0, x1) = (1, 3): 1 + 6 = 7
    # At (x0, x1) = (2, 4): 2 + 8 = 10
    # Second output: 3*x0 + x1
    # At (x0, x1) = (1, 3): 3 + 3 = 6
    # At (x0, x1) = (2, 4): 6 + 4 = 10
    assert torch.allclose(lower, torch.tensor([7.0, 6.0]))
    assert torch.allclose(upper, torch.tensor([10.0, 10.0]))


def test_matmul_2d_constant() -> None:
    """Test matmul with 2D weight matrix."""
    # Region: x0 ∈ [0, 1], x1 ∈ [0, 1]
    # Matmul: [x0, x1] @ [[2, 0], [0, 3]] = [2*x0, 3*x1]
    region = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([1.0, 1.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds, weight)

    lower, upper = result.concretize()
    # Expected: [2*x0, 3*x1]
    # At (x0, x1) = (0, 0): [0, 0]
    # At (x0, x1) = (1, 1): [2, 3]
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 3.0]))


def test_matmul_negative_weights() -> None:
    """Test matmul with negative weights."""
    # Region: x0 ∈ [1, 2]
    # Matmul: [[1], [-2]] @ [x0] = [x0, -2*x0]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0], [-2.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, weight, bounds)

    lower, upper = result.concretize()
    # Expected: [x0, -2*x0]
    # At x0=1: [1, -2]
    # At x0=2: [2, -4]
    assert torch.allclose(lower, torch.tensor([1.0, -4.0]))
    assert torch.allclose(upper, torch.tensor([2.0, -2.0]))
