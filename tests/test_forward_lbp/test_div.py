from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.div import ForwardLBPDiv
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


def _make_identity_bounds_preserve_shape(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds while preserving the region tensor shape."""
    shape = tuple(region.lower.shape)
    in_features = region.lower.numel()
    identity = torch.eye(in_features).reshape(*shape, *shape)
    zero_bias = torch.zeros_like(region.lower)
    return LinearBounds(
        regions=[region],
        linear_lower=[identity],
        bias_lower=zero_bias,
        linear_upper=[identity],
        bias_upper=zero_bias,
    )


def test_div_abstract_abstract_positive() -> None:
    """Test division of two abstract positive intervals."""
    # Region: x0 ∈ [6, 12], x1 ∈ [2, 3]
    # Division: x0 / x1
    # Result: [6/3, 12/2] = [2, 6]
    region = HyperRectangle(lower=torch.tensor([6.0, 2.0]), upper=torch.tensor([12.0, 3.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds_a, bounds_b)

    # Division loses linear dependency
    assert result.linear_lowers == []
    assert result.linear_uppers == []
    assert torch.allclose(result.bias_lower, torch.tensor([2.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([6.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([6.0]))


def test_div_abstract_constant_positive() -> None:
    """Test division of abstract by positive constant."""
    # Region: x ∈ [4, 8]
    # Division: x / 2
    # Result: [2, 4]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([8.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds, torch.tensor(2.0))

    # Should preserve linear structure: x/2
    assert result.linear_lower is not None
    assert result.linear_upper is not None
    assert torch.allclose(result.linear_lower, torch.tensor([[0.5]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.5]]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))


def test_div_abstract_constant_negative() -> None:
    """Test division of abstract by negative constant."""
    # Region: x ∈ [4, 8]
    # Division: x / (-2)
    # Result: [-4, -2]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([8.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds, torch.tensor(-2.0))

    # Linear bounds should be flipped: x/(-2) = -x/2
    assert torch.allclose(result.linear_lower, torch.tensor([[-0.5]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-0.5]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-4.0]))
    assert torch.allclose(upper, torch.tensor([-2.0]))


def test_div_crossing_zero_divisor() -> None:
    """Test division when divisor crosses zero (unbounded)."""
    # Region: x0 ∈ [4, 8], x1 ∈ [-1, 1]
    # Division: x0 / x1 (x1 crosses zero)
    # Result: [-inf, inf]
    region = HyperRectangle(lower=torch.tensor([4.0, -1.0]), upper=torch.tensor([8.0, 1.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds_a, bounds_b)

    lower, upper = result.concretize()
    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()


def test_div_crossing_zero_divisor_is_elementwise() -> None:
    """Only outputs whose divisor interval crosses zero should be unbounded."""
    # Region: x0 ∈ [6, 12], x1 ∈ [-1, 1], x2 ∈ [2, 3]
    # Divisions: [x0/x1, x0/x2]
    # Result: [(-inf, inf), (2, 6)]
    region = HyperRectangle(lower=torch.tensor([6.0, -1.0, 2.0]), upper=torch.tensor([12.0, 1.0, 3.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds_a, bounds_b)

    lower, upper = result.concretize()

    assert torch.isneginf(lower[0])
    assert torch.isposinf(upper[0])
    assert torch.allclose(lower[1], torch.tensor(2.0))
    assert torch.allclose(upper[1], torch.tensor(6.0))


def test_constant_div_crossing_zero_divisor_is_elementwise() -> None:
    """For constant/abstract division, only zero-crossing denominator outputs should be unbounded."""
    # Region: x0 ∈ [-1, 1], x1 ∈ [2, 3]
    # Divisions: [6/x0, 6/x1]
    # Result: [(-inf, inf), (2, 3)]
    region = HyperRectangle(lower=torch.tensor([-1.0, 2.0]), upper=torch.tensor([1.0, 3.0]))

    denominator_bounds = _make_linear_bounds(region)
    strategy = ForwardLBPDiv()
    result = propagate(strategy, torch.tensor(6.0), denominator_bounds)

    lower, upper = result.concretize()

    assert torch.isneginf(lower[0])
    assert torch.isposinf(upper[0])
    assert torch.isfinite(lower[1])
    assert torch.isfinite(upper[1])
    assert lower[1].item() <= 2.0 + 1e-6
    assert upper[1].item() >= 3.0 - 1e-6


def test_constant_div_negative_constant_flips_bounds() -> None:
    """Negative numerator should flip reciprocal lower/upper orientations."""
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([4.0]))

    denominator_bounds = _make_linear_bounds(region)
    strategy = ForwardLBPDiv()
    result = propagate(strategy, torch.tensor(-6.0), denominator_bounds)

    # For -6/x over x in [2, 4], reciprocal relaxation parameters give:
    # lower: (3/4) x - 9/2
    # upper: (2/3) x - 4
    assert torch.allclose(result.linear_lower, torch.tensor([[0.75]]))
    assert torch.allclose(result.bias_lower, torch.tensor([-4.5]))
    assert torch.allclose(result.linear_upper, torch.tensor([[2.0 / 3.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([-4.0]))

    lower, upper = result.concretize()
    assert lower.item() <= -3.0 + 1e-6
    assert upper.item() >= -1.5 - 1e-6
    assert torch.all(lower <= upper)


def test_constant_div_zero_constant_is_exact_zero() -> None:
    """0/x should produce exact zero bounds even when denominator crosses zero."""
    region = HyperRectangle(lower=torch.tensor([-1.0, 2.0]), upper=torch.tensor([1.0, 3.0]))

    denominator_bounds = _make_linear_bounds(region)
    strategy = ForwardLBPDiv()
    result = propagate(strategy, torch.tensor(0.0), denominator_bounds)

    # 0/x is identically zero even if denominator interval crosses zero.
    assert torch.allclose(result.linear_lower, torch.zeros_like(result.linear_lower))
    assert torch.allclose(result.bias_lower, torch.zeros_like(result.bias_lower))
    assert torch.allclose(result.linear_upper, torch.zeros_like(result.linear_upper))
    assert torch.allclose(result.bias_upper, torch.zeros_like(result.bias_upper))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.zeros_like(lower))
    assert torch.allclose(upper, torch.zeros_like(upper))


def test_div_abstract_batch_constant_broadcasted() -> None:
    """Test abstract / constant with batched constants broadcast across trailing axes."""
    region = HyperRectangle(
        lower=torch.tensor([[4.0, 8.0], [2.0, 6.0]]),
        upper=torch.tensor([[8.0, 12.0], [6.0, 10.0]]),
    )
    bounds = _make_identity_bounds_preserve_shape(region)
    batch_constants = torch.tensor([[2.0], [-2.0]])

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds, batch_constants)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([[2.0, 4.0], [-3.0, -5.0]]))
    assert torch.allclose(upper, torch.tensor([[4.0, 6.0], [-1.0, -3.0]]))


def test_constant_div_batch_constant_nominal() -> None:
    """Test constant / abstract with batched constants on strictly positive denominators."""
    region = HyperRectangle(
        lower=torch.tensor([[2.0, 4.0], [2.0, 4.0]]),
        upper=torch.tensor([[4.0, 8.0], [4.0, 8.0]]),
    )
    denominator_bounds = _make_identity_bounds_preserve_shape(region)
    batch_constants = torch.tensor([[8.0], [-8.0]])

    strategy = ForwardLBPDiv()
    result = propagate(strategy, batch_constants, denominator_bounds)

    lower, upper = result.concretize()
    assert torch.isfinite(lower).all()
    assert torch.isfinite(upper).all()

    # Row 0: 8/x -> [2, 4] and [1, 2]
    assert lower[0, 0].item() <= 2.0 + 1e-6
    assert upper[0, 0].item() >= 4.0 - 1e-6
    assert lower[0, 1].item() <= 1.0 + 1e-6
    assert upper[0, 1].item() >= 2.0 - 1e-6

    # Row 1: -8/x -> [-4, -2] and [-2, -1]
    assert lower[1, 0].item() <= -4.0 + 1e-6
    assert upper[1, 0].item() >= -2.0 - 1e-6
    assert lower[1, 1].item() <= -2.0 + 1e-6
    assert upper[1, 1].item() >= -1.0 - 1e-6


def test_constant_div_batch_constant_mixed_zero_crossing() -> None:
    """Test element-wise unbounded behavior for batched constants with mixed denominator regimes."""
    region = HyperRectangle(
        lower=torch.tensor([[-1.0, 2.0], [2.0, -1.0]]),
        upper=torch.tensor([[1.0, 4.0], [4.0, 1.0]]),
    )
    denominator_bounds = _make_identity_bounds_preserve_shape(region)
    batch_constants = torch.tensor([[6.0], [-6.0]])

    strategy = ForwardLBPDiv()
    result = propagate(strategy, batch_constants, denominator_bounds)

    lower, upper = result.concretize()

    # Cross-zero elements are unbounded.
    assert torch.isneginf(lower[0, 0])
    assert torch.isposinf(upper[0, 0])
    assert torch.isneginf(lower[1, 1])
    assert torch.isposinf(upper[1, 1])

    # Non-crossing elements remain finite and sound.
    assert torch.isfinite(lower[0, 1])
    assert torch.isfinite(upper[0, 1])
    assert lower[0, 1].item() <= 1.5 + 1e-6
    assert upper[0, 1].item() >= 3.0 - 1e-6

    assert torch.isfinite(lower[1, 0])
    assert torch.isfinite(upper[1, 0])
    assert lower[1, 0].item() <= -3.0 + 1e-6
    assert upper[1, 0].item() >= -1.5 - 1e-6
