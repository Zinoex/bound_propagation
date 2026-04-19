"""Alpha-CROWN resolvers for each relaxation.

Strategies call these helpers to obtain an optimizable alpha override from
the current :class:`AlphaProvider`. Each resolver:

1. Computes a per-element init tensor that reproduces the op's analytical
   default in the regime where the knob is active. Elements outside the
   active regime receive a safe placeholder (any value in ``[0, 1]``;
   they are never consumed by the relaxation function).
2. Asks the provider for a unit-interval fraction tensor of matching
   shape. Returns ``None`` when the provider opts out (e.g. when alpha
   optimization is disabled for this node).

The returned tensor is passed straight through to the corresponding
``compute_*_relaxation`` function via its ``alpha_<knob>`` keyword.
"""

from __future__ import annotations

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from ..alpha_optimization import AlphaProvider


def _safe_z(lower: torch.Tensor, upper: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return ``u / (u - l)`` clamped to ``[0, 1]`` with zero-denominator guard.

    In the ReLU/abs crossing regime (``l < 0 < u``) this evaluates to the
    ratio used as the analytical default lower slope; outside that regime
    the result is a safe placeholder that never gets consumed by the
    relaxation function.
    """
    denom = (upper - lower).clamp(min=eps)
    return (upper / denom).clamp(0.0, 1.0)


def resolve_relu_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the optional alpha-CROWN override for a ReLU node.

    In the crossing regime the lower-slope fraction directly maps to the
    slope value in ``[0, 1]``. The init tensor reproduces the non-adaptive
    analytical default ``z = u / (u - l)``.
    """
    lower = input_bounds.lower
    upper = input_bounds.upper
    crossing = (lower < 0) & (upper > 0)
    z = _safe_z(lower, upper)
    init = torch.where(crossing, z, 0.5)
    return provider.get(
        node=node,
        knob_name="relu_lower_slope",
        shape=lower.shape,
        init=init.detach(),
        device=lower.device,
        dtype=lower.dtype,
    )


def resolve_abs_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the optional alpha-CROWN override for an abs node.

    In the crossing regime the fraction ``alpha`` maps to lower-bound slope
    ``m = 2 * alpha - 1 in [-1, 1]``. The init tensor places the slope at
    the current default ``m = (u + l) / (u - l)``, i.e. fraction
    ``alpha = u / (u - l)``.
    """
    lower = input_bounds.lower
    upper = input_bounds.upper
    crossing = (lower < 0) & (upper > 0)
    z = _safe_z(lower, upper)
    init = torch.where(crossing, z, 0.5)
    return provider.get(
        node=node,
        knob_name="abs_lower_slope",
        shape=lower.shape,
        init=init.detach(),
        device=lower.device,
        dtype=lower.dtype,
    )


def _resolve_tangent_pair(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
    knob_lower: str,
    knob_upper: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve a pair of midpoint-default tangent-point fractions.

    Both knobs default to the midpoint fraction ``0.5``. One controls the
    lower-bound tangent (active on convex subintervals), the other controls
    the upper-bound tangent (active on concave subintervals). Relaxations
    that use this pair are responsible for applying the override only in the
    regime where the tangent is a sound bound for every ``alpha in [0, 1]``.
    """
    lower = input_bounds.lower
    alpha_lo = provider.get(
        node=node,
        knob_name=knob_lower,
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )
    alpha_up = provider.get(
        node=node,
        knob_name=knob_upper,
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )
    return alpha_lo, alpha_up


def _resolve_eta_pair(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
    knob_lower: str,
    knob_upper: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve two [0, 1] eta knobs (default 0.5 each) for a paired-op node."""
    lower = reference.lower
    eta_lo = provider.get(
        node=node,
        knob_name=knob_lower,
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )
    eta_up = provider.get(
        node=node,
        knob_name=knob_upper,
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )
    return eta_lo, eta_up


def resolve_mul_etas(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve McCormick eta fractions for multiplication (default 0.5)."""
    return _resolve_eta_pair(provider, node, reference, "mul_eta_lower", "mul_eta_upper")


def resolve_matmul_etas(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve McCormick eta fractions for matmul (default 0.5).

    The ``reference`` bounds have shape ``(*batch, M, K, N)``: one bilinear
    term ``a_ik * b_kj`` per entry. One optimizable knob is allocated per
    term, mirroring the per-element knob layout used for element-wise
    multiplication.
    """
    return _resolve_eta_pair(provider, node, reference, "matmul_eta_lower", "matmul_eta_upper")


def resolve_div_etas(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve McCormick eta fractions for division (default 0.5)."""
    return _resolve_eta_pair(provider, node, reference, "div_eta_lower", "div_eta_upper")


def resolve_max_etas(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve eta fractions for element-wise maximum (default 0.5)."""
    return _resolve_eta_pair(provider, node, reference, "max_eta_lower", "max_eta_upper")


def resolve_min_etas(
    provider: AlphaProvider,
    node: fx.Node,
    reference: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve eta fractions for element-wise minimum (default 0.5)."""
    return _resolve_eta_pair(provider, node, reference, "min_eta_lower", "min_eta_upper")


def _resolve_single_midpoint_knob(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
    knob_name: str,
) -> torch.Tensor | None:
    """Resolve a single ``[0, 1]`` knob with midpoint default (0.5)."""
    lower = input_bounds.lower
    return provider.get(
        node=node,
        knob_name=knob_name,
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )


def resolve_sin_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the sin tangent-point fraction (default 0.5, safe subregime only)."""
    return _resolve_single_midpoint_knob(provider, node, input_bounds, "sin_tangent_frac")


def resolve_cos_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the cos tangent-point fraction (default 0.5, safe subregime only)."""
    return _resolve_single_midpoint_knob(provider, node, input_bounds, "cos_tangent_frac")


def resolve_tan_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the tan tangent-point fraction (default 0.5, asymptote-free branch)."""
    return _resolve_single_midpoint_knob(provider, node, input_bounds, "tan_tangent_frac")


def resolve_exp_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the exp lower-bound tangent-point fraction (default 0.5)."""
    lower = input_bounds.lower
    return provider.get(
        node=node,
        knob_name="exp_tangent_lower",
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )


def resolve_log_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the log upper-bound tangent-point fraction (default 0.5)."""
    lower = input_bounds.lower
    return provider.get(
        node=node,
        knob_name="log_tangent_upper",
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )


def resolve_sqrt_alpha(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> torch.Tensor | None:
    """Resolve the sqrt upper-bound tangent-point fraction (default 0.5)."""
    lower = input_bounds.lower
    return provider.get(
        node=node,
        knob_name="sqrt_tangent_upper",
        shape=lower.shape,
        init=0.5,
        device=lower.device,
        dtype=lower.dtype,
    )


def resolve_reciprocal_alphas(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve reciprocal tangent-point fractions.

    Returns ``(lower_branch, upper_branch)``. The lower-branch knob is only
    consumed on strictly-positive inputs (convex regime, controls the
    lower-bound tangent). The upper-branch knob is only consumed on
    strictly-negative inputs (concave regime, controls the upper-bound
    tangent). The crossing-zero regime remains ``[-inf, +inf]`` and ignores
    both overrides.
    """
    return _resolve_tangent_pair(
        provider,
        node,
        input_bounds,
        knob_lower="reciprocal_tangent_lower",
        knob_upper="reciprocal_tangent_upper",
    )


def resolve_sigmoid_alphas(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve sigmoid tangent-point overrides (lower, upper).

    The lower-bound tangent is only consumed in the negative-only regime
    (``u <= 0``, sigmoid is convex there); the upper-bound tangent is only
    consumed in the positive-only regime (``l >= 0``, sigmoid is concave
    there). The crossing regime falls back to its regime-split analytical
    logic and ignores both overrides.
    """
    return _resolve_tangent_pair(
        provider,
        node,
        input_bounds,
        knob_lower="sigmoid_tangent_lower",
        knob_upper="sigmoid_tangent_upper",
    )


def resolve_tanh_alphas(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve tanh tangent-point overrides (lower, upper). See sigmoid."""
    return _resolve_tangent_pair(
        provider,
        node,
        input_bounds,
        knob_lower="tanh_tangent_lower",
        knob_upper="tanh_tangent_upper",
    )


def resolve_clamp_alphas(
    provider: AlphaProvider,
    node: fx.Node,
    input_bounds: IntervalBounds,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve the two optional alpha-CROWN overrides for a clamp node.

    The first knob controls the lower-bound slope in the ``crosses_min``
    regime; the second controls the upper-bound slope in the
    ``crosses_max`` regime. Both are fractions in ``[0, 1]`` mapped
    directly to slopes in ``[0, 1]``. Default fraction ``0`` reproduces
    the current horizontal-line bounds.
    """
    lower = input_bounds.lower
    alpha_lower = provider.get(
        node=node,
        knob_name="clamp_crosses_min_lower_slope",
        shape=lower.shape,
        init=0.0,
        device=lower.device,
        dtype=lower.dtype,
    )
    alpha_upper = provider.get(
        node=node,
        knob_name="clamp_crosses_max_upper_slope",
        shape=lower.shape,
        init=0.0,
        device=lower.device,
        dtype=lower.dtype,
    )
    return alpha_lower, alpha_upper
