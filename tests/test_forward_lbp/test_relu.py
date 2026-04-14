from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.relu import ForwardLBPRelu
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


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


def test_relu_positive_interval() -> None:
    """Test ReLU on a positive interval (inactive relaxation)."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x, upper = x
    # ReLU([2, 5]) = [2, 5] (identity since all positive)
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    # Linear should be preserved (identity) since all positive
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_relu_negative_interval() -> None:
    """Test ReLU on a negative interval (all zeros)."""
    # Region: x ∈ [-5, -2]
    # Bounds: lower = x, upper = x
    # ReLU([-5, -2]) = [0, 0]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    # The implementation keeps an explicit zero linear map rather than dropping
    # it to None; both encode the same concretized interval [0, 0].
    assert result.linear_lower is not None
    assert result.linear_upper is not None
    assert torch.allclose(result.linear_lower, torch.zeros_like(result.linear_lower))
    assert torch.allclose(result.linear_upper, torch.zeros_like(result.linear_upper))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([0.0]))


def test_relu_crossing_interval() -> None:
    """Test ReLU on an interval crossing zero (relaxation needed)."""
    # Region: x ∈ [-2, 3]
    # Bounds: lower = x, upper = x
    # ReLU([-2, 3]) = [0, 3]
    # Lower bound: 0 (inactive portion)
    # Upper bound: linear relaxation from (-2, 0) to (3, 3)
    # Slope: (3 - 0) / (3 - (-2)) = 3/5 = 0.6
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    # Lower/upper both use alpha=u/(u-l)=0.6 in non-adaptive mode.
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    # The linear_upper coefficient should be the slope 3/5 = 0.6
    assert torch.allclose(result.linear_upper, torch.tensor([[0.6]]), atol=1e-5)
    assert torch.allclose(result.linear_lower, torch.tensor([[0.6]]), atol=1e-5)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-1.2]))
    assert torch.allclose(upper, torch.tensor([3.0]))


def test_relu_zero_boundary() -> None:
    """Test ReLU on an interval starting at zero."""
    # Region: x ∈ [0, 4]
    # Bounds: lower = x, upper = x
    # ReLU([0, 4]) = [0, 4]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    # Should preserve identity (all non-negative)
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))


def test_relu_with_bias() -> None:
    """Test ReLU with non-identity linear bounds."""
    # Region: x ∈ [0, 2]
    # Bounds: lower = 2x - 3, upper = 2x + 1
    # At x=0: input ∈ [-3, 1], ReLU ∈ [0, 1]
    # At x=2: input ∈ [1, 5], ReLU ∈ [1, 5]
    # Conservative interval: [0, 5]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[2.0]]),
        bias_lower=torch.tensor([-3.0]),
        linear_upper=torch.tensor([[2.0]]),
        bias_upper=torch.tensor([1.0]),
    )

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # With non-adaptive crossing relaxation the lower bound is sound but can be
    # below the exact ReLU lower endpoint.
    assert torch.all(lower <= torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([5.0]), atol=1e-5)


def test_relu_multidimensional() -> None:
    """Test ReLU on multidimensional bounds."""
    # Region: x0 ∈ [-1, 1], x1 ∈ [2, 4], x2 ∈ [-3, -1]
    # Bounds: identity
    # ReLU: ([0, 1], [2, 4], [0, 0])
    region = HyperRectangle(lower=torch.tensor([-1.0, 2.0, -3.0]), upper=torch.tensor([1.0, 4.0, -1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPRelu()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # Element 0: ReLU([-1, 1]) = [0, 1]
    assert torch.allclose(lower[0], torch.tensor(-0.5))
    assert torch.allclose(upper[0], torch.tensor(1.0))
    # Element 1: ReLU([2, 4]) = [2, 4]
    assert torch.allclose(lower[1], torch.tensor(2.0))
    assert torch.allclose(upper[1], torch.tensor(4.0))
    # Element 2: ReLU([-3, -1]) = [0, 0]
    assert torch.allclose(lower[2], torch.tensor(0.0))
    assert torch.allclose(upper[2], torch.tensor(0.0))
