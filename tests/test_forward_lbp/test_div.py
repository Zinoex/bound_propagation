from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.pairwise import ForwardLBPDiv
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

    W_l = result.linear_lowers[0]  # shape (1, 2)
    b_l = result.bias_lower
    W_u = result.linear_uppers[0]
    b_u = result.bias_upper

    # Evaluate the linear bounds at the four corners and verify both the exact
    # predicted values and that they form a valid lower/upper bound on x0/x1.
    # eta=0.5 selects the McCormick convex-combination midpoint; the upper relaxation
    # is tight at (x0=6, x1=3) and (x0=12, x1=2).
    for x0, x1, true_val, exp_lower, exp_upper in [
        (6.0, 2.0, 3.0, 2.82, 3.50),
        (6.0, 3.0, 2.0, 1.38, 2.00),  # upper tight here
        (12.0, 2.0, 6.0, 5.32, 6.00),  # upper tight here
        (12.0, 3.0, 4.0, 3.88, 4.50),
    ]:
        x = torch.tensor([x0, x1])
        lower_val = (W_l @ x + b_l).item()
        upper_val = (W_u @ x + b_u).item()
        assert abs(lower_val - exp_lower) < 1e-3, f"lower at ({x0},{x1}): got {lower_val}, expected {exp_lower}"
        assert abs(upper_val - exp_upper) < 1e-3, f"upper at ({x0},{x1}): got {upper_val}, expected {exp_upper}"
        assert lower_val <= true_val + 1e-5, f"lower bound not sound at ({x0},{x1})"
        assert upper_val >= true_val - 1e-5, f"upper bound not sound at ({x0},{x1})"


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

    W_l = result.linear_lowers[0]  # shape (1, 2)
    b_l = result.bias_lower
    W_u = result.linear_uppers[0]
    b_u = result.bias_upper

    # All linear coefficients are zeroed and bias = ±∞, so the bound evaluates
    # to [-∞, +∞] at every input point.
    for x in [torch.tensor([4.0, -1.0]), torch.tensor([8.0, 0.5]), torch.tensor([6.0, 0.0])]:
        assert (W_l @ x + b_l).isneginf().all()
        assert (W_u @ x + b_u).isposinf().all()


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

    W_l = result.linear_lowers[0]  # shape (2, 3)
    b_l = result.bias_lower
    W_u = result.linear_uppers[0]
    b_u = result.bias_upper

    # Output 0 (x1 crosses zero): linear coefficients are zeroed, bias = ±∞.
    for x in [torch.tensor([9.0, -1.0, 2.5]), torch.tensor([6.0, 0.0, 3.0])]:
        assert (W_l @ x + b_l)[0].isneginf()
        assert (W_u @ x + b_u)[0].isposinf()

    # Output 1 (x2 ∈ [2, 3]): McCormick via reciprocal relaxation.
    # The x1 coefficient is zero, so only x0 and x2 matter.
    # eta=0.5 makes the upper tight at (x0=6, x2=3) and (x0=12, x2=2).
    for x0, x2, true_val, exp_lower, exp_upper in [
        (6.0, 3.0, 2.0, 1.38, 2.00),  # upper tight here
        (12.0, 2.0, 6.0, 5.32, 6.00),  # upper tight here
        (12.0, 3.0, 4.0, 3.88, 4.50),
    ]:
        x = torch.tensor([x0, 0.0, x2])
        lower_val = (W_l @ x + b_l)[1].item()
        upper_val = (W_u @ x + b_u)[1].item()
        assert abs(lower_val - exp_lower) < 1e-3, f"lower at ({x0},{x2}): got {lower_val}, expected {exp_lower}"
        assert abs(upper_val - exp_upper) < 1e-3, f"upper at ({x0},{x2}): got {upper_val}, expected {exp_upper}"
        assert lower_val <= true_val + 1e-5, f"lower bound not sound at ({x0},{x2})"
        assert upper_val >= true_val - 1e-5, f"upper bound not sound at ({x0},{x2})"


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
