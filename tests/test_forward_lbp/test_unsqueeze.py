from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.unsqueeze import ForwardLBPUnsqueeze
from bound_propagation.regions import HyperRectangle

from tests.helpers import propagate

def test_unsqueeze_dim0() -> None:
    """Test unsqueeze at dimension 0."""
    # Region: x ∈ [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (12,) flattened from conceptual (3, 4)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(12, 1),
        bias_lower=torch.arange(12.0),
        linear_upper=torch.ones(12, 1),
        bias_upper=torch.arange(12.0) + 1,
    )

    strategy = ForwardLBPUnsqueeze()
    result = propagate(strategy, bounds, dim=0)

    # Should add a dimension at position 0: (12,) -> (1, 12)
    assert result.bias_lower.shape == (1, 12)
    assert result.bias_upper.shape == (1, 12)

def test_unsqueeze_middle_dim() -> None:
    """Test unsqueeze at a middle dimension."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (6,) flattened from conceptual (2, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(6, 1),
        bias_lower=torch.arange(6.0),
        linear_upper=torch.ones(6, 1),
        bias_upper=torch.arange(6.0) + 1,
    )

    strategy = ForwardLBPUnsqueeze()
    result = propagate(strategy, bounds, dim=0)

    # Should add a dimension at position 0: (6,) -> (1, 6)
    assert result.bias_lower.shape == (1, 6)
    assert result.bias_upper.shape == (1, 6)

def test_unsqueeze_last_dim() -> None:
    """Test unsqueeze at the last dimension."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (3,)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(3, 1),
        bias_lower=torch.arange(3.0),
        linear_upper=torch.ones(3, 1),
        bias_upper=torch.arange(3.0) + 1,
    )

    strategy = ForwardLBPUnsqueeze()
    result = propagate(strategy, bounds, dim=1)

    # Should add a dimension at the end: (3,) -> (3, 1)
    assert result.bias_lower.shape == (3, 1)
    assert result.bias_upper.shape == (3, 1)

def test_unsqueeze_1d_to_2d() -> None:
    """Test unsqueeze converting 1D to 2D tensor."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

    # Shape: (5,)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.ones(5, 1),
        bias_lower=torch.arange(5.0),
        linear_upper=torch.ones(5, 1),
        bias_upper=torch.arange(5.0) + 1,
    )

    strategy = ForwardLBPUnsqueeze()
    result = propagate(strategy, bounds, dim=0)

    # Should convert to (1, 5)
    assert result.bias_lower.shape == (1, 5)
    assert result.bias_upper.shape == (1, 5)
    assert torch.allclose(result.bias_lower, torch.arange(5.0).view(1, 5))
    assert torch.allclose(result.bias_upper, (torch.arange(5.0) + 1).view(1, 5))
