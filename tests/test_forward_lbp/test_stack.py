from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.stack import ForwardLBPStack
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def test_stack_two_tensors() -> None:
    """Test stacking two tensors."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
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

    strategy = ForwardLBPStack()
    result = propagate(strategy, [bounds1, bounds2], dim=0)

    # Result should stack [x0, x1]
    lower, upper = result.concretize()
    # torch.stack([tensor([1.]), tensor([3.])], dim=0) -> tensor([[1.], [3.]])
    assert torch.allclose(lower, torch.tensor([[1.0], [3.0]]))
    assert torch.allclose(upper, torch.tensor([[2.0], [4.0]]))


def test_stack_three_tensors() -> None:
    """Test stacking three tensors."""
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

    strategy = ForwardLBPStack()
    result = propagate(strategy, [bounds1, bounds2, bounds3], dim=0)

    lower, upper = result.concretize()
    # torch.stack([tensor([1.]), tensor([2.]), tensor([3.])], dim=0) -> tensor([[1.], [2.], [3.]])
    assert torch.allclose(lower, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.allclose(upper, torch.tensor([[2.0], [4.0], [5.0]]))


def test_stack_dim1() -> None:
    """Test stacking along dimension 1."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Both tensors have shape (2,)
    bounds1 = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 1),
        bias_lower=torch.tensor([1.0, 2.0]),
        linear_upper=torch.ones(2, 1),
        bias_upper=torch.tensor([2.0, 3.0]),
    )

    bounds2 = LinearBounds(
        region=region,
        linear_lower=torch.ones(2, 1),
        bias_lower=torch.tensor([3.0, 4.0]),
        linear_upper=torch.ones(2, 1),
        bias_upper=torch.tensor([4.0, 5.0]),
    )

    strategy = ForwardLBPStack()
    result = propagate(strategy, [bounds1, bounds2], dim=1)

    # Result should be (2, 2)
    lower, upper = result.concretize()
    assert lower.shape == (2, 2)
    assert upper.shape == (2, 2)
    assert torch.allclose(lower, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
