from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.shape import ForwardLBPGetItem
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def test_getitem_single_index() -> None:
    """Test getitem with a single index."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.arange(5.0),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.arange(5.0) + 1,
    )

    strategy = ForwardLBPGetItem()
    result = propagate(strategy, bounds, 2)

    # getitem preserves affine coefficients; with x in [0, 1] and coefficient 1,
    # each selected upper gains +1 from the input contribution.
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor(2.0))
    assert torch.allclose(upper, torch.tensor(4.0))


def test_getitem_slice() -> None:
    """Test getitem with a slice."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.arange(5.0),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.arange(5.0) + 1,
    )

    strategy = ForwardLBPGetItem()
    result = propagate(strategy, bounds, slice(1, 4))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(upper, torch.tensor([3.0, 4.0, 5.0]))


def test_getitem_2d_single_index() -> None:
    """Test getitem on 2D tensor with single index."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(3, 4, 1),
        bias_lower=torch.arange(12.0).view(3, 4),
        linear_upper=torch.ones(3, 4, 1),
        bias_upper=torch.arange(12.0).view(3, 4) + 1,
    )

    strategy = ForwardLBPGetItem()
    result = propagate(strategy, bounds, 1)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.allclose(upper, torch.tensor([6.0, 7.0, 8.0, 9.0]))


def test_getitem_with_trailing_ellipsis() -> None:
    """Test getitem with a trailing ellipsis in the index."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(2, 3, 4, 1),
        bias_lower=torch.arange(24.0).view(2, 3, 4),
        linear_upper=torch.ones(2, 3, 4, 1),
        bias_upper=torch.arange(24.0).view(2, 3, 4) + 1,
    )

    strategy = ForwardLBPGetItem()
    result = propagate(strategy, bounds, (..., 2))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([[2.0, 6.0, 10.0], [14.0, 18.0, 22.0]]))
    assert torch.allclose(upper, torch.tensor([[4.0, 8.0, 12.0], [16.0, 20.0, 24.0]]))


def test_getitem_with_middle_ellipsis() -> None:
    """Test getitem with an ellipsis between explicit indices."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        linear_lower=torch.ones(2, 3, 4, 1),
        bias_lower=torch.arange(24.0).view(2, 3, 4),
        linear_upper=torch.ones(2, 3, 4, 1),
        bias_upper=torch.arange(24.0).view(2, 3, 4) + 1,
    )

    strategy = ForwardLBPGetItem()
    result = propagate(strategy, bounds, (1, ..., 2))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([14.0, 18.0, 22.0]))
    assert torch.allclose(upper, torch.tensor([16.0, 20.0, 24.0]))
