from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.shape import ForwardLBPReshape
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def test_reshape_1d_to_2d() -> None:
    """Test reshaping 1D tensor to 2D."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(6, 1),
        bias_lower=torch.arange(6.0),
        linear_upper=torch.ones(6, 1),
        bias_upper=torch.arange(6.0) + 1,
    )

    strategy = ForwardLBPReshape()
    result = propagate(strategy, bounds, 2, 3)

    assert result.bias_lower.shape == (2, 3)
    assert result.bias_upper.shape == (2, 3)
    assert torch.allclose(result.bias_lower, torch.arange(6.0).view(2, 3))
    assert torch.allclose(result.bias_upper, (torch.arange(6.0) + 1).view(2, 3))


def test_reshape_2d_to_1d() -> None:
    """Test reshaping 2D tensor to 1D."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(2, 3, 1),
        bias_lower=torch.arange(6.0).view(2, 3),
        linear_upper=torch.ones(2, 3, 1),
        bias_upper=(torch.arange(6.0) + 1).view(2, 3),
    )

    strategy = ForwardLBPReshape()
    result = propagate(strategy, bounds, 6)

    assert result.bias_lower.shape == (6,)
    assert result.bias_upper.shape == (6,)
    assert torch.allclose(result.bias_lower, torch.arange(6.0))
    assert torch.allclose(result.bias_upper, torch.arange(6.0) + 1)


def test_reshape_3d() -> None:
    """Test reshaping between 3D shapes."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    bounds = LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.ones(2, 2, 3, 1),
        bias_lower=torch.arange(12.0).view(2, 2, 3),
        linear_upper=torch.ones(2, 2, 3, 1),
        bias_upper=(torch.arange(12.0) + 1).view(2, 2, 3),
    )

    strategy = ForwardLBPReshape()
    result = propagate(strategy, bounds, 3, 4)

    assert result.bias_lower.shape == (3, 4)
    assert result.bias_upper.shape == (3, 4)
    assert torch.allclose(result.bias_lower, torch.arange(12.0).view(3, 4))
    assert torch.allclose(result.bias_upper, (torch.arange(12.0) + 1).view(3, 4))
