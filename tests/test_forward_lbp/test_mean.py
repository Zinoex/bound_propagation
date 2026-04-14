from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.mean import ForwardLBPMean
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle, shape: tuple[int, ...]) -> LinearBounds:
    """Create identity linear bounds from a region with specific shape."""
    dim = region.lower.numel()
    flat_dim = int(torch.prod(torch.tensor(shape)).item())

    # Create identity for flattened view
    linear = torch.eye(dim)
    # Reshape to match output shape if needed
    if flat_dim == dim:
        linear_reshaped = linear.view(*shape, dim)
    else:
        # For reduction tests, we'll use simpler linear forms
        linear_reshaped = linear[:flat_dim].view(*shape, dim)

    return LinearBounds(
        region=region,
        linear_lower=linear_reshaped,
        bias_lower=torch.zeros(shape),
        linear_upper=linear_reshaped,
        bias_upper=torch.zeros(shape),
    )


def test_mean_along_last_dim() -> None:
    """Test mean reduction along the last dimension."""
    # Region: x ∈ [0, 2] for 6 elements shaped (2, 3)
    # Bounds: identity
    # mean along dim=-1: average of each row
    # Row 0: mean([0, 2], [0, 2], [0, 2]) = [0, 2]
    # Row 1: mean([0, 2], [0, 2], [0, 2]) = [0, 2]
    region = HyperRectangle(
        lower=torch.zeros(6),
        upper=torch.full((6,), 2.0),
    )

    # Create linear bounds for shape (2, 3)
    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(6).view(2, 3, 6),
        bias_lower=torch.zeros(2, 3),
        linear_upper=torch.eye(6).view(2, 3, 6),
        bias_upper=torch.zeros(2, 3),
    )

    # Mean along last dimension (dim=-1 or dim=1)
    strategy = ForwardLBPMean()

    # Create mock node with dim attribute

    result = propagate(strategy, bounds, dim=-1, keepdim=False)

    # After mean along dim=-1, shape should be (2,)
    # Result concretizes
    assert result.linear_lower is None
    assert result.linear_upper is None

    lower, upper = result.concretize()
    # Each row averages to the same interval [0, 2]
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 2.0]))


def test_mean_all_elements() -> None:
    """Test mean over all elements (no dim specified)."""
    # Region: x ∈ [1, 3] for 4 elements
    # Bounds: identity
    # mean(): average of all elements = [1, 3]
    region = HyperRectangle(
        lower=torch.ones(4),
        upper=torch.full((4,), 3.0),
    )

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(4),
        bias_lower=torch.zeros(4),
        linear_upper=torch.eye(4),
        bias_upper=torch.zeros(4),
    )

    strategy = ForwardLBPMean()

    result = propagate(strategy, bounds)

    # Concretizes to scalar
    lower, upper = result.concretize()
    # Should be scalar or 1-element tensor
    assert lower.numel() == 1
    assert upper.numel() == 1
    # Mean of [1, 3] across all elements is [1, 3]
    assert torch.allclose(lower, torch.tensor(1.0))
    assert torch.allclose(upper, torch.tensor(3.0))


def test_mean_with_keepdim() -> None:
    """Test mean with keepdim=True."""
    # Region: x ∈ [0, 4] for shape (2, 3)
    # Bounds: identity
    # mean(dim=1, keepdim=True): shape (2, 1)
    region = HyperRectangle(
        lower=torch.zeros(6),
        upper=torch.full((6,), 4.0),
    )

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.eye(6).view(2, 3, 6),
        bias_lower=torch.zeros(2, 3),
        linear_upper=torch.eye(6).view(2, 3, 6),
        bias_upper=torch.zeros(2, 3),
    )

    strategy = ForwardLBPMean()

    result = propagate(strategy, bounds, dim=1, keepdim=True)

    lower, upper = result.concretize()
    # Shape should be (2, 1) with keepdim
    assert lower.shape == (2, 1)
    assert upper.shape == (2, 1)
    assert torch.allclose(lower, torch.tensor([[0.0], [0.0]]))
    assert torch.allclose(upper, torch.tensor([[4.0], [4.0]]))
