"""Per-alpha soundness sweeps for relaxations with optimizable knobs.

For every op that accepts an ``alpha_<knob>`` override, we sweep
``alpha in {0.0, 0.1, ..., 1.0}`` across every regime (positive-only,
negative-only, crossing, zero-width, ...) and verify that the resulting
linear relaxation bounds the true function at 2000 sampled points.

This is the primary acceptance gate for alpha-CROWN correctness: if any
cell fails, the override is unsound and the plan's guarantees break.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.linear_relaxations.elementwise import (
    ElementwiseParams,
    compute_abs_relaxation,
    compute_clamp_relaxation,
    compute_cos_relaxation,
    compute_exp_relaxation,
    compute_log_relaxation,
    compute_reciprocal_relaxation,
    compute_relu_relaxation,
    compute_sigmoid_relaxation,
    compute_sin_relaxation,
    compute_sqrt_relaxation,
    compute_tan_relaxation,
    compute_tanh_relaxation,
)
from bound_propagation.propagation.linear_relaxations.pairwise import (
    PairedParams,
    compute_div_relaxation,
    compute_maximum_relaxation,
    compute_minimum_relaxation,
    compute_mul_relaxation,
)

ALPHA_GRID: list[float] = [i / 10 for i in range(11)]
NUM_SAMPLES = 2000
ATOL = 1e-5


def _assert_elementwise_sound(
    fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: IntervalBounds,
    params: ElementwiseParams,
    num_samples: int = NUM_SAMPLES,
    atol: float = ATOL,
) -> None:
    """Verify ``alpha_lower*x + beta_lower <= fn(x) <= alpha_upper*x + beta_upper``.

    Samples ``num_samples`` points uniformly over ``[bounds.lower, bounds.upper]``
    and checks soundness at every point to the given absolute tolerance.
    """
    lower = bounds.lower
    upper = bounds.upper

    t = torch.linspace(0.0, 1.0, num_samples, dtype=lower.dtype, device=lower.device)
    t = t.view(-1, *([1] * lower.ndim))
    samples = lower + t * (upper - lower)

    true_vals = fn(samples)
    lower_pred = params.alpha_lower * samples + params.beta_lower
    upper_pred = params.alpha_upper * samples + params.beta_upper

    lo_violations = torch.nonzero(lower_pred > true_vals + atol, as_tuple=False)
    if lo_violations.numel() > 0:
        max_gap = (lower_pred - true_vals).max().item()
        raise AssertionError(f"Lower bound violated at {lo_violations.shape[0]} samples; max violation = {max_gap:.6g}")

    up_violations = torch.nonzero(upper_pred < true_vals - atol, as_tuple=False)
    if up_violations.numel() > 0:
        max_gap = (true_vals - upper_pred).max().item()
        raise AssertionError(f"Upper bound violated at {up_violations.shape[0]} samples; max violation = {max_gap:.6g}")


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


_RELU_REGIMES: dict[str, tuple[float, float]] = {
    "positive_only": (1.0, 3.0),
    "negative_only": (-3.0, -1.0),
    "crossing_balanced": (-2.0, 2.0),
    "crossing_positive_heavy": (-1.0, 5.0),
    "crossing_negative_heavy": (-5.0, 1.0),
    "zero_width_positive": (2.0, 2.0),
    "zero_width_negative": (-2.0, -2.0),
    "zero_width_at_zero": (0.0, 0.0),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_RELU_REGIMES.keys()))
def test_relu_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _RELU_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_relu_relaxation(bounds, adaptive=False, alpha_relu_lower=override)
    _assert_elementwise_sound(torch.relu, bounds, params)


def test_relu_alpha_none_matches_default() -> None:
    """Passing ``alpha_relu_lower=None`` must reproduce the current default exactly."""
    lower = torch.tensor([-2.5])
    upper = torch.tensor([1.5])
    bounds = IntervalBounds(lower, upper)
    default = compute_relu_relaxation(bounds, adaptive=False)
    z = upper / (upper - lower)  # fraction reproducing current non-adaptive default
    override = torch.where(
        (lower < 0) & (upper > 0),
        z,
        torch.full_like(lower, 0.5),
    )
    via_override = compute_relu_relaxation(bounds, adaptive=False, alpha_relu_lower=override)
    for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
        assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), attr


# ---------------------------------------------------------------------------
# Abs
# ---------------------------------------------------------------------------


_ABS_REGIMES: dict[str, tuple[float, float]] = {
    "positive_only": (0.5, 4.0),
    "negative_only": (-4.0, -0.5),
    "crossing_balanced": (-2.0, 2.0),
    "crossing_positive_heavy": (-1.0, 4.0),
    "crossing_negative_heavy": (-4.0, 1.0),
    "zero_width_positive": (2.0, 2.0),
    "zero_width_negative": (-2.0, -2.0),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_ABS_REGIMES.keys()))
def test_abs_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _ABS_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_abs_relaxation(bounds, alpha_abs_lower=override)
    _assert_elementwise_sound(torch.abs, bounds, params)


def test_abs_alpha_none_matches_default() -> None:
    lower = torch.tensor([-2.5])
    upper = torch.tensor([1.5])
    bounds = IntervalBounds(lower, upper)
    default = compute_abs_relaxation(bounds)
    # Default fraction = (slope_default + 1) / 2 = u / (u - l) in crossing regime.
    z = upper / (upper - lower)
    override = torch.where(
        (lower < 0) & (upper > 0),
        z,
        torch.full_like(lower, 0.5),
    )
    via_override = compute_abs_relaxation(bounds, alpha_abs_lower=override)
    for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
        assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), attr


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


# Clamp regimes (input interval vs. [min_val, max_val]):
# The alpha knobs are only active in `crosses_min` (lower-bound slope) and
# `crosses_max` (upper-bound slope); all other regimes must ignore the override.
_CLAMP_REGIMES: dict[str, tuple[float, float, float, float]] = {
    # (lower, upper, min_val, max_val)
    "in_range": (-0.5, 0.5, -1.0, 1.0),
    "below_min": (-3.0, -2.0, -1.0, 1.0),
    "above_max": (2.0, 3.0, -1.0, 1.0),
    "crosses_min": (-2.0, 0.5, -1.0, 1.0),
    "crosses_max": (0.5, 2.0, -1.0, 1.0),
    "crosses_both": (-2.0, 2.0, -1.0, 1.0),
    "zero_width_in_range": (0.5, 0.5, -1.0, 1.0),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_CLAMP_REGIMES.keys()))
def test_clamp_alpha_sound(regime: str, alpha: float) -> None:
    lo, up, lo_clamp, up_clamp = _CLAMP_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_clamp_relaxation(
        bounds,
        min_val=lo_clamp,
        max_val=up_clamp,
        alpha_clamp_crosses_min_lower=override,
        alpha_clamp_crosses_max_upper=override,
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=lo_clamp, max=up_clamp)

    _assert_elementwise_sound(fn, bounds, params)


def test_clamp_alpha_zero_matches_default() -> None:
    """Alpha fraction 0 reproduces the horizontal default for both active knobs."""
    lower = torch.tensor([-2.0, 0.5])
    upper = torch.tensor([0.5, 2.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_clamp_relaxation(bounds, min_val=-1.0, max_val=1.0)
    zeros = torch.zeros_like(lower)
    via_override = compute_clamp_relaxation(
        bounds,
        min_val=-1.0,
        max_val=1.0,
        alpha_clamp_crosses_min_lower=zeros,
        alpha_clamp_crosses_max_upper=zeros,
    )
    for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
        assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), attr


# ---------------------------------------------------------------------------
# Sigmoid and Tanh (single-regime tangent-point knobs)
# ---------------------------------------------------------------------------


_SIGMOID_REGIMES: dict[str, tuple[float, float]] = {
    "negative_only": (-4.0, -1.0),
    "negative_only_narrow": (-2.0, -0.5),
    "positive_only": (0.5, 3.0),
    "positive_only_wide": (1.0, 5.0),
    "crossing_balanced": (-2.0, 2.0),
    "crossing_positive_heavy": (-1.0, 4.0),
    "crossing_negative_heavy": (-4.0, 1.0),
    "zero_width_positive": (1.5, 1.5),
    "zero_width_negative": (-1.5, -1.5),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_SIGMOID_REGIMES.keys()))
def test_sigmoid_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _SIGMOID_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_sigmoid_relaxation(
        bounds,
        alpha_sigmoid_tangent_lower=override,
        alpha_sigmoid_tangent_upper=override,
    )
    _assert_elementwise_sound(torch.sigmoid, bounds, params)


def test_sigmoid_alpha_half_matches_default() -> None:
    """Alpha fraction 0.5 (midpoint) reproduces the analytical default."""
    lower = torch.tensor([-3.0, 0.5, -1.0])
    upper = torch.tensor([-0.5, 2.5, 1.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_sigmoid_relaxation(bounds)
    halves = torch.full_like(lower, 0.5)
    via_override = compute_sigmoid_relaxation(
        bounds,
        alpha_sigmoid_tangent_lower=halves,
        alpha_sigmoid_tangent_upper=halves,
    )
    for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
        assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), attr


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_SIGMOID_REGIMES.keys()))
def test_tanh_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _SIGMOID_REGIMES[regime]  # reuse regime grid
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_tanh_relaxation(
        bounds,
        alpha_tanh_tangent_lower=override,
        alpha_tanh_tangent_upper=override,
    )
    _assert_elementwise_sound(torch.tanh, bounds, params)


def test_tanh_alpha_half_matches_default() -> None:
    lower = torch.tensor([-3.0, 0.5, -1.0])
    upper = torch.tensor([-0.5, 2.5, 1.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_tanh_relaxation(bounds)
    halves = torch.full_like(lower, 0.5)
    via_override = compute_tanh_relaxation(
        bounds,
        alpha_tanh_tangent_lower=halves,
        alpha_tanh_tangent_upper=halves,
    )
    for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
        assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), attr


def test_sigmoid_alpha_crossing_regime_ignores_override() -> None:
    """In the crossing regime any override must be silently ignored."""
    lower = torch.tensor([-2.0])
    upper = torch.tensor([2.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_sigmoid_relaxation(bounds)
    for alpha in ALPHA_GRID:
        override = torch.full_like(lower, alpha)
        via_override = compute_sigmoid_relaxation(
            bounds,
            alpha_sigmoid_tangent_lower=override,
            alpha_sigmoid_tangent_upper=override,
        )
        for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
            assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), (
                f"crossing regime: override alpha={alpha} changed {attr}"
            )


def test_tanh_alpha_crossing_regime_ignores_override() -> None:
    lower = torch.tensor([-2.0])
    upper = torch.tensor([2.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_tanh_relaxation(bounds)
    for alpha in ALPHA_GRID:
        override = torch.full_like(lower, alpha)
        via_override = compute_tanh_relaxation(
            bounds,
            alpha_tanh_tangent_lower=override,
            alpha_tanh_tangent_upper=override,
        )
        for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
            assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), (
                f"crossing regime: override alpha={alpha} changed {attr}"
            )


# ---------------------------------------------------------------------------
# Exp, Log, Sqrt, Reciprocal (globally convex/concave tangent-point knobs)
# ---------------------------------------------------------------------------


_EXP_REGIMES: dict[str, tuple[float, float]] = {
    "negative": (-3.0, -0.5),
    "crossing": (-1.0, 2.0),
    "positive": (0.5, 3.0),
    "wide": (-4.0, 4.0),
    "zero_width": (0.5, 0.5),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_EXP_REGIMES.keys()))
def test_exp_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _EXP_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_exp_relaxation(bounds, alpha_exp_tangent_lower=override)
    _assert_elementwise_sound(torch.exp, bounds, params)


_LOG_REGIMES: dict[str, tuple[float, float]] = {
    "small_positive": (0.1, 0.9),
    "around_one": (0.5, 2.5),
    "large": (2.0, 10.0),
    "zero_width": (1.5, 1.5),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_LOG_REGIMES.keys()))
def test_log_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _LOG_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_log_relaxation(bounds, alpha_log_tangent_upper=override)
    _assert_elementwise_sound(torch.log, bounds, params)


_SQRT_REGIMES: dict[str, tuple[float, float]] = {
    "small": (0.01, 0.5),
    "medium": (0.5, 4.0),
    "large": (4.0, 16.0),
    "zero_width": (2.25, 2.25),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_SQRT_REGIMES.keys()))
def test_sqrt_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _SQRT_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_sqrt_relaxation(bounds, alpha_sqrt_tangent_upper=override)
    _assert_elementwise_sound(torch.sqrt, bounds, params)


_RECIPROCAL_REGIMES: dict[str, tuple[float, float]] = {
    "all_positive_small": (0.5, 2.0),
    "all_positive_wide": (0.1, 5.0),
    "all_negative_small": (-2.0, -0.5),
    "all_negative_wide": (-5.0, -0.1),
    "zero_width_positive": (1.0, 1.0),
    "zero_width_negative": (-1.0, -1.0),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_RECIPROCAL_REGIMES.keys()))
def test_reciprocal_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _RECIPROCAL_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_reciprocal_relaxation(
        bounds,
        alpha_reciprocal_tangent_lower=override,
        alpha_reciprocal_tangent_upper=override,
    )
    _assert_elementwise_sound(torch.reciprocal, bounds, params)


def test_reciprocal_crossing_zero_ignores_override() -> None:
    """Reciprocal on a zero-crossing interval is undefined; overrides must not change that."""
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    bounds = IntervalBounds(lower, upper)
    default = compute_reciprocal_relaxation(bounds)
    for alpha in ALPHA_GRID:
        override = torch.full_like(lower, alpha)
        via_override = compute_reciprocal_relaxation(
            bounds,
            alpha_reciprocal_tangent_lower=override,
            alpha_reciprocal_tangent_upper=override,
        )
        for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
            assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7, equal_nan=True), (
                f"crossing-zero regime: override alpha={alpha} changed {attr}"
            )


def test_exp_log_sqrt_alpha_half_matches_default() -> None:
    lower_exp = torch.tensor([-1.0, 0.5])
    upper_exp = torch.tensor([2.0, 3.0])
    bounds_exp = IntervalBounds(lower_exp, upper_exp)
    halves = torch.full_like(lower_exp, 0.5)
    assert torch.allclose(
        compute_exp_relaxation(bounds_exp).alpha_lower,
        compute_exp_relaxation(bounds_exp, alpha_exp_tangent_lower=halves).alpha_lower,
        atol=1e-7,
    )

    lower_log = torch.tensor([0.5, 2.0])
    upper_log = torch.tensor([2.0, 5.0])
    bounds_log = IntervalBounds(lower_log, upper_log)
    halves_log = torch.full_like(lower_log, 0.5)
    assert torch.allclose(
        compute_log_relaxation(bounds_log).alpha_upper,
        compute_log_relaxation(bounds_log, alpha_log_tangent_upper=halves_log).alpha_upper,
        atol=1e-7,
    )

    lower_sqrt = torch.tensor([0.5, 2.0])
    upper_sqrt = torch.tensor([2.0, 5.0])
    bounds_sqrt = IntervalBounds(lower_sqrt, upper_sqrt)
    halves_sqrt = torch.full_like(lower_sqrt, 0.5)
    assert torch.allclose(
        compute_sqrt_relaxation(bounds_sqrt).alpha_upper,
        compute_sqrt_relaxation(bounds_sqrt, alpha_sqrt_tangent_upper=halves_sqrt).alpha_upper,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Sin, Cos, Tan (safe-subregime tangent-point knobs + fallback soundness)
# ---------------------------------------------------------------------------

# Each tuple covers multiple regimes: safe subintervals (single-convex/concave,
# no extremum, no inflection) where the knob is active, plus regimes where the
# knob should be silently ignored but the result must remain sound.
_SIN_REGIMES: dict[str, tuple[float, float]] = {
    "safe_concave_ascending": (0.1, 1.0),  # [0, π/2] — concave, increasing
    "safe_concave_descending": (math.pi / 2 + 0.1, math.pi - 0.1),
    "safe_convex_descending": (math.pi + 0.1, 3 * math.pi / 2 - 0.1),
    "safe_convex_ascending": (3 * math.pi / 2 + 0.1, 2 * math.pi - 0.1),
    "has_max_no_min": (0.0, math.pi),
    "has_min_no_max": (math.pi, 2 * math.pi),
    "crosses_inflection_no_extrema": (math.pi - 0.3, math.pi + 0.3),
    "has_both_extrema": (0.0, 2 * math.pi),
    "zero_width": (1.0, 1.0),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_SIN_REGIMES.keys()))
def test_sin_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _SIN_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_sin_relaxation(bounds, alpha_sin_tangent_frac=override)
    _assert_elementwise_sound(torch.sin, bounds, params)


_COS_REGIMES: dict[str, tuple[float, float]] = {
    "safe_concave_right": (-math.pi / 2 + 0.1, 0.0),  # concave, increasing on [-π/2, 0]
    "safe_concave_left": (0.0, math.pi / 2 - 0.1),  # concave, decreasing on [0, π/2]
    "safe_convex_right": (math.pi / 2 + 0.1, math.pi),
    "safe_convex_left": (math.pi, 3 * math.pi / 2 - 0.1),
    "has_max_no_min": (-math.pi / 2, math.pi / 2),
    "has_min_no_max": (math.pi / 2, 3 * math.pi / 2),
    "has_both_extrema": (-math.pi / 2, 3 * math.pi / 2),
    "zero_width": (0.5, 0.5),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_COS_REGIMES.keys()))
def test_cos_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _COS_REGIMES[regime]
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_cos_relaxation(bounds, alpha_cos_tangent_frac=override)
    _assert_elementwise_sound(torch.cos, bounds, params)


_TAN_REGIMES: dict[str, tuple[float, float]] = {
    "safe_convex_pos": (0.1, math.pi / 2 - 0.2),
    "safe_concave_neg": (-math.pi / 2 + 0.2, -0.1),
    "crosses_zero_inflection": (-0.4, 0.4),
    "crosses_asymptote": (math.pi / 2 - 0.1, math.pi / 2 + 0.1),
    "zero_width": (0.25, 0.25),
}


@pytest.mark.parametrize("alpha", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_TAN_REGIMES.keys()))
def test_tan_alpha_sound(regime: str, alpha: float) -> None:
    lo, up = _TAN_REGIMES[regime]
    if regime == "crosses_asymptote":
        # In the asymptote-crossing regime the relaxation sets +/-inf bounds;
        # soundness is trivial and Monte Carlo would sample near the asymptote
        # where true values blow up. Skip to avoid spurious violations.
        lower = torch.tensor([lo])
        upper = torch.tensor([up])
        bounds = IntervalBounds(lower, upper)
        override = torch.full_like(lower, alpha)
        params = compute_tan_relaxation(bounds, alpha_tan_tangent_frac=override)
        assert torch.isinf(params.beta_lower).all()
        assert torch.isinf(params.beta_upper).all()
        return
    lower = torch.tensor([lo])
    upper = torch.tensor([up])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, alpha)
    params = compute_tan_relaxation(bounds, alpha_tan_tangent_frac=override)
    _assert_elementwise_sound(torch.tan, bounds, params)


def test_sin_cos_unsafe_regime_ignores_override() -> None:
    """On multi-extremum / inflection-crossing intervals the override must be ignored."""
    # has_both_extrema for sin: default is constant [-1, 1]; override should not change it.
    lower = torch.tensor([0.0])
    upper = torch.tensor([2 * math.pi])
    bounds = IntervalBounds(lower, upper)
    default = compute_sin_relaxation(bounds)
    for alpha in ALPHA_GRID:
        override = torch.full_like(lower, alpha)
        via_override = compute_sin_relaxation(bounds, alpha_sin_tangent_frac=override)
        for attr in ("alpha_lower", "beta_lower", "alpha_upper", "beta_upper"):
            assert torch.allclose(getattr(default, attr), getattr(via_override, attr), atol=1e-7), (
                f"has_both_extrema: alpha={alpha} changed sin {attr}"
            )


# ---------------------------------------------------------------------------
# Paired ops: mul, div, max, min (McCormick-style eta knobs)
# ---------------------------------------------------------------------------


def _assert_paired_sound(
    fn,
    bounds_a: IntervalBounds,
    bounds_b: IntervalBounds,
    params: PairedParams,
    num_samples: int = NUM_SAMPLES,
    atol: float = ATOL,
) -> None:
    """Sample (a, b) uniformly in their intervals and verify the paired relaxation."""
    la, ua = bounds_a.lower, bounds_a.upper
    lb, ub = bounds_b.lower, bounds_b.upper

    ta = torch.linspace(0.0, 1.0, num_samples, dtype=la.dtype, device=la.device)
    ta = ta.view(-1, *([1] * la.ndim))
    samples_a = la + ta * (ua - la)
    tb = torch.linspace(0.0, 1.0, num_samples, dtype=lb.dtype, device=lb.device).flip(0)
    tb = tb.view(-1, *([1] * lb.ndim))
    samples_b = lb + tb * (ub - lb)

    true_vals = fn(samples_a, samples_b)
    lower_pred = params.alpha_lower_a * samples_a + params.alpha_lower_b * samples_b + params.bias_lower
    upper_pred = params.alpha_upper_a * samples_a + params.alpha_upper_b * samples_b + params.bias_upper

    if torch.any(lower_pred > true_vals + atol):
        max_gap = (lower_pred - true_vals).max().item()
        raise AssertionError(f"Paired lower bound violated; max violation = {max_gap:.6g}")
    if torch.any(upper_pred < true_vals - atol):
        max_gap = (true_vals - upper_pred).max().item()
        raise AssertionError(f"Paired upper bound violated; max violation = {max_gap:.6g}")


_MUL_REGIMES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "a_pos_b_pos": ((0.5, 2.0), (1.0, 3.0)),
    "a_neg_b_neg": ((-2.0, -0.5), (-3.0, -1.0)),
    "a_pos_b_neg": ((0.5, 2.0), (-3.0, -1.0)),
    "a_cross_b_pos": ((-1.0, 2.0), (0.5, 3.0)),
    "a_cross_b_cross": ((-1.0, 2.0), (-2.0, 1.0)),
}


@pytest.mark.parametrize("eta", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_MUL_REGIMES.keys()))
def test_mul_eta_sound(regime: str, eta: float) -> None:
    (la, ua), (lb, ub) = _MUL_REGIMES[regime]
    lower_a = torch.tensor([la])
    upper_a = torch.tensor([ua])
    lower_b = torch.tensor([lb])
    upper_b = torch.tensor([ub])
    bounds_a = IntervalBounds(lower_a, upper_a)
    bounds_b = IntervalBounds(lower_b, upper_b)
    eta_t = torch.full_like(lower_a, eta)
    params = compute_mul_relaxation(bounds_a, bounds_b, eta_lower=eta_t, eta_upper=eta_t)
    _assert_paired_sound(lambda a, b: a * b, bounds_a, bounds_b, params)


_DIV_REGIMES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "a_pos_b_pos": ((0.5, 2.0), (1.0, 3.0)),
    "a_neg_b_pos": ((-2.0, -0.5), (1.0, 3.0)),
    "a_cross_b_neg": ((-1.0, 2.0), (-3.0, -1.0)),
}


@pytest.mark.parametrize("eta", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_DIV_REGIMES.keys()))
def test_div_eta_sound(regime: str, eta: float) -> None:
    (la, ua), (lb, ub) = _DIV_REGIMES[regime]
    lower_a = torch.tensor([la])
    upper_a = torch.tensor([ua])
    lower_b = torch.tensor([lb])
    upper_b = torch.tensor([ub])
    bounds_a = IntervalBounds(lower_a, upper_a)
    bounds_b = IntervalBounds(lower_b, upper_b)
    eta_t = torch.full_like(lower_a, eta)
    params = compute_div_relaxation(bounds_a, bounds_b, eta_lower=eta_t, eta_upper=eta_t)
    _assert_paired_sound(lambda a, b: a / b, bounds_a, bounds_b, params)


_MAXMIN_REGIMES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "a_dominates": ((5.0, 8.0), (1.0, 3.0)),
    "b_dominates": ((1.0, 3.0), (5.0, 8.0)),
    "crossing_overlap": ((1.0, 5.0), (3.0, 7.0)),
    "crossing_tight": ((-1.0, 1.0), (-1.0, 1.0)),
}


@pytest.mark.parametrize("eta", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_MAXMIN_REGIMES.keys()))
def test_max_eta_sound(regime: str, eta: float) -> None:
    (la, ua), (lb, ub) = _MAXMIN_REGIMES[regime]
    bounds_a = IntervalBounds(torch.tensor([la]), torch.tensor([ua]))
    bounds_b = IntervalBounds(torch.tensor([lb]), torch.tensor([ub]))
    eta_t = torch.full_like(bounds_a.lower, eta)
    params = compute_maximum_relaxation(bounds_a, bounds_b, eta_lower=eta_t, eta_upper=eta_t)
    _assert_paired_sound(lambda a, b: torch.maximum(a, b), bounds_a, bounds_b, params)


@pytest.mark.parametrize("eta", ALPHA_GRID)
@pytest.mark.parametrize("regime", list(_MAXMIN_REGIMES.keys()))
def test_min_eta_sound(regime: str, eta: float) -> None:
    (la, ua), (lb, ub) = _MAXMIN_REGIMES[regime]
    bounds_a = IntervalBounds(torch.tensor([la]), torch.tensor([ua]))
    bounds_b = IntervalBounds(torch.tensor([lb]), torch.tensor([ub]))
    eta_t = torch.full_like(bounds_a.lower, eta)
    params = compute_minimum_relaxation(bounds_a, bounds_b, eta_lower=eta_t, eta_upper=eta_t)
    _assert_paired_sound(lambda a, b: torch.minimum(a, b), bounds_a, bounds_b, params)


def test_mul_eta_tensor_matches_scalar_half() -> None:
    """Tensor eta filled with 0.5 must match passing the scalar 0.5."""
    bounds_a = IntervalBounds(torch.tensor([-1.0, 1.0]), torch.tensor([2.0, 3.0]))
    bounds_b = IntervalBounds(torch.tensor([-2.0, 0.5]), torch.tensor([1.0, 2.5]))
    scalar = compute_mul_relaxation(bounds_a, bounds_b)
    halves = torch.full_like(bounds_a.lower, 0.5)
    tensor = compute_mul_relaxation(bounds_a, bounds_b, eta_lower=halves, eta_upper=halves)
    for attr in ("alpha_lower_a", "alpha_upper_a", "alpha_lower_b", "alpha_upper_b", "bias_lower", "bias_upper"):
        assert torch.allclose(getattr(scalar, attr), getattr(tensor, attr), atol=1e-7), attr


# ---------------------------------------------------------------------------
# Autograd smoke test: gradients flow from output params into alpha overrides.
# ---------------------------------------------------------------------------


def test_relu_alpha_requires_grad_propagates() -> None:
    lower = torch.tensor([-2.0, -1.0, 1.0])
    upper = torch.tensor([2.0, 1.0, 3.0])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, 0.5, dtype=torch.float32).requires_grad_(True)
    params = compute_relu_relaxation(bounds, adaptive=False, alpha_relu_lower=override)
    params.alpha_lower.sum().backward()
    assert override.grad is not None
    # Gradient is non-zero only at crossing elements (indices 0 and 1).
    assert override.grad[0].item() == pytest.approx(1.0)
    assert override.grad[1].item() == pytest.approx(1.0)
    assert override.grad[2].item() == pytest.approx(0.0)


def test_abs_alpha_requires_grad_propagates() -> None:
    lower = torch.tensor([-2.0, 1.0])
    upper = torch.tensor([2.0, 3.0])
    bounds = IntervalBounds(lower, upper)
    override = torch.full_like(lower, 0.5, dtype=torch.float32).requires_grad_(True)
    params = compute_abs_relaxation(bounds, alpha_abs_lower=override)
    params.alpha_lower.sum().backward()
    assert override.grad is not None
    # d/dalpha (2*alpha - 1) = 2 in crossing regime, 0 elsewhere.
    assert override.grad[0].item() == pytest.approx(2.0)
    assert override.grad[1].item() == pytest.approx(0.0)


def test_clamp_alpha_requires_grad_propagates() -> None:
    lower = torch.tensor([-2.0, 0.5])
    upper = torch.tensor([0.5, 2.0])
    bounds = IntervalBounds(lower, upper)
    override_lower = torch.full_like(lower, 0.5, dtype=torch.float32).requires_grad_(True)
    override_upper = torch.full_like(lower, 0.5, dtype=torch.float32).requires_grad_(True)
    params = compute_clamp_relaxation(
        bounds,
        min_val=-1.0,
        max_val=1.0,
        alpha_clamp_crosses_min_lower=override_lower,
        alpha_clamp_crosses_max_upper=override_upper,
    )
    (params.alpha_lower.sum() + params.alpha_upper.sum()).backward()
    # Lower override only contributes in crosses_min (index 0).
    assert override_lower.grad is not None
    assert override_lower.grad[0].item() == pytest.approx(1.0)
    assert override_lower.grad[1].item() == pytest.approx(0.0)
    # Upper override only contributes in crosses_max (index 1).
    assert override_upper.grad is not None
    assert override_upper.grad[0].item() == pytest.approx(0.0)
    assert override_upper.grad[1].item() == pytest.approx(1.0)
