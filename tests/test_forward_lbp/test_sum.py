from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.reduction import ForwardLBPSum
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def test_sum_all_elements() -> None:
    """Test sum over all elements."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    strategy = ForwardLBPSum()
    result = propagate(strategy, bounds, dim=None, keepdim=False)

    # Sum of [1, 2, 3, 4, 5] to [3, 4, 5, 6, 7] is [15, 25]
    # (concretize gives: lower = 0*ones + [1,2,3,4,5] = [1,2,3,4,5]
    #                    upper = 1*ones + [2,3,4,5,6] = [3,4,5,6,7])
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor(15.0))
    assert torch.allclose(upper, torch.tensor(25.0))


def test_sum_along_dim() -> None:
    """Test sum along a specific dimension."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (6,) flattened from conceptual (2, 3)
    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(6, 1),
        bias_lower=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        linear_upper=torch.ones(6, 1),
        bias_upper=torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    )

    strategy = ForwardLBPSum()
    result = propagate(strategy, bounds, dim=None, keepdim=False)

    # Concretize: lower = [1,2,3,4,5,6], upper = [3,4,5,6,7,8]
    # Sum all: lower = 21, upper = 33
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor(21.0))
    assert torch.allclose(upper, torch.tensor(33.0))


def test_sum_keepdim() -> None:
    """Test sum with keepdim=True."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(3, 1),
        bias_lower=torch.tensor([1.0, 2.0, 3.0]),
        linear_upper=torch.ones(3, 1),
        bias_upper=torch.tensor([2.0, 3.0, 4.0]),
    )

    strategy = ForwardLBPSum()
    result = propagate(strategy, bounds, dim=0, keepdim=True)

    # Concretize: lower = [1,2,3], upper = [3,4,5]
    # Sum with keepdim: lower = [6], upper = [12]
    lower, upper = result.concretize()
    assert lower.shape == (1,)
    assert upper.shape == (1,)
    assert torch.allclose(lower, torch.tensor([6.0]))
    assert torch.allclose(upper, torch.tensor([12.0]))
