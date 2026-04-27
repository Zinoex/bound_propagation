"""Specs that bundle a ``compute_*_relaxation`` with its α-resolver(s).

Each :class:`ElementwiseSpec` is a single source of truth for one element-wise
activation: which math function to call, which α-CROWN resolver(s) feed it, and
how to extract any extra kwargs (e.g. ``power`` for ``pow``, ``min``/``max`` for
``clamp``) from the fx node.

Forward and backward LBP both consume these specs via thin generic strategy
classes (see ``forward_lbp/elementwise.py`` and ``backward_lbp/elementwise.py``);
this avoids 26 nearly-identical wrapper classes that differed only in which
``compute_*`` and ``resolve_*`` they called.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from ..alpha_optimization import AlphaProvider
from .alpha_resolvers import (
    resolve_abs_alpha,
    resolve_clamp_alphas,
    resolve_cos_alpha,
    resolve_exp_alpha,
    resolve_log_alpha,
    resolve_pow_alpha,
    resolve_reciprocal_alphas,
    resolve_relu_alpha,
    resolve_sigmoid_alphas,
    resolve_sin_alpha,
    resolve_sqrt_alpha,
    resolve_tan_alpha,
    resolve_tanh_alphas,
)
from .elementwise import (
    ElementwiseParams,
    compute_abs_relaxation,
    compute_clamp_relaxation,
    compute_cos_relaxation,
    compute_exp_relaxation,
    compute_log_relaxation,
    compute_pow_relaxation,
    compute_reciprocal_relaxation,
    compute_relu_relaxation,
    compute_sigmoid_relaxation,
    compute_sin_relaxation,
    compute_sqrt_relaxation,
    compute_tan_relaxation,
    compute_tanh_relaxation,
)

AlphaResolver = Callable[[AlphaProvider, fx.Node, IntervalBounds], dict[str, torch.Tensor | None]]
ExtraKwargsExtractor = Callable[[fx.Node, tuple, dict], dict[str, Any]]


def _no_extra(_node: fx.Node, _args: tuple, _kwargs: dict) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ElementwiseSpec:
    """How to compute one element-wise relaxation given input bounds and α.

    Attributes
    ----------
    name :
        Short identifier (matches the registry key, e.g. ``"relu"``). Used in
        error messages.
    compute :
        The ``compute_*_relaxation`` function to invoke. Takes
        ``(input_bounds, **alpha_kwargs, **extra_kwargs)`` and returns
        :class:`ElementwiseParams`.
    resolve_alpha :
        Builds the α-kwargs dict for this op. Returns ``{}``-equivalent
        (every value ``None``) when α is disabled or absent.
    extract_extra :
        Pulls extra positional/keyword args from the fx node into kwargs for
        ``compute`` (e.g. ``power`` for pow, ``min``/``max`` for clamp).
        Defaults to no extra args.
    """

    name: str
    compute: Callable[..., ElementwiseParams]
    resolve_alpha: AlphaResolver
    extract_extra: ExtraKwargsExtractor = field(default=_no_extra)

    def build_params(
        self,
        input_bounds: IntervalBounds,
        node: fx.Node,
        provider: AlphaProvider,
        args: tuple,
        kwargs: dict,
    ) -> ElementwiseParams:
        """Run the full pipeline: extract extras → resolve α → call ``compute``."""
        extra = self.extract_extra(node, args, kwargs)
        alphas = self.resolve_alpha(provider, node, input_bounds)
        return self.compute(input_bounds, **alphas, **extra)


# ---------------------------------------------------------------------------
# Per-op resolver wrappers (uniform dict-returning shape)
# ---------------------------------------------------------------------------


def _relu_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_relu_lower": resolve_relu_alpha(provider, node, bounds)}


def _abs_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_abs_lower": resolve_abs_alpha(provider, node, bounds)}


def _exp_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_exp_tangent_lower": resolve_exp_alpha(provider, node, bounds)}


def _log_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_log_tangent_upper": resolve_log_alpha(provider, node, bounds)}


def _sqrt_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_sqrt_tangent_upper": resolve_sqrt_alpha(provider, node, bounds)}


def _sin_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_sin_tangent_frac": resolve_sin_alpha(provider, node, bounds)}


def _cos_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_cos_tangent_frac": resolve_cos_alpha(provider, node, bounds)}


def _tan_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_tan_tangent_frac": resolve_tan_alpha(provider, node, bounds)}


def _pow_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    return {"alpha_pow_tangent": resolve_pow_alpha(provider, node, bounds)}


def _reciprocal_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    lo, up = resolve_reciprocal_alphas(provider, node, bounds)
    return {"alpha_reciprocal_tangent_lower": lo, "alpha_reciprocal_tangent_upper": up}


def _sigmoid_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    lo, up = resolve_sigmoid_alphas(provider, node, bounds)
    return {"alpha_sigmoid_tangent_lower": lo, "alpha_sigmoid_tangent_upper": up}


def _tanh_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    lo, up = resolve_tanh_alphas(provider, node, bounds)
    return {"alpha_tanh_tangent_lower": lo, "alpha_tanh_tangent_upper": up}


def _clamp_alphas(provider: AlphaProvider, node: fx.Node, bounds: IntervalBounds) -> dict[str, Any]:
    cm_lo, cmx_up = resolve_clamp_alphas(provider, node, bounds)
    return {"alpha_clamp_crosses_min_lower": cm_lo, "alpha_clamp_crosses_max_upper": cmx_up}


# ---------------------------------------------------------------------------
# Per-op extra-kwargs extractors
# ---------------------------------------------------------------------------


def _pow_extra(_node: fx.Node, args: tuple, _kwargs: dict) -> dict[str, Any]:
    """``torch.pow(x, n)`` — second positional is the integer exponent."""
    if len(args) < 2:
        raise ValueError("pow requires a power argument")
    power = args[1]
    if not isinstance(power, int):
        raise TypeError(f"pow requires an int exponent, got {type(power).__name__}")
    return {"power": power}


def _clamp_extra(_node: fx.Node, args: tuple, kwargs: dict) -> dict[str, Any]:
    """``torch.clamp(x, min=..., max=...)`` — pull from positional or keyword."""
    return {
        "min_val": args[1] if len(args) > 1 else kwargs.get("min"),
        "max_val": args[2] if len(args) > 2 else kwargs.get("max"),
    }


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------


RELU_SPEC = ElementwiseSpec("relu", compute_relu_relaxation, _relu_alphas)
ABS_SPEC = ElementwiseSpec("abs", compute_abs_relaxation, _abs_alphas)
EXP_SPEC = ElementwiseSpec("exp", compute_exp_relaxation, _exp_alphas)
LOG_SPEC = ElementwiseSpec("log", compute_log_relaxation, _log_alphas)
SQRT_SPEC = ElementwiseSpec("sqrt", compute_sqrt_relaxation, _sqrt_alphas)
SIN_SPEC = ElementwiseSpec("sin", compute_sin_relaxation, _sin_alphas)
COS_SPEC = ElementwiseSpec("cos", compute_cos_relaxation, _cos_alphas)
TAN_SPEC = ElementwiseSpec("tan", compute_tan_relaxation, _tan_alphas)
RECIPROCAL_SPEC = ElementwiseSpec("reciprocal", compute_reciprocal_relaxation, _reciprocal_alphas)
SIGMOID_SPEC = ElementwiseSpec("sigmoid", compute_sigmoid_relaxation, _sigmoid_alphas)
TANH_SPEC = ElementwiseSpec("tanh", compute_tanh_relaxation, _tanh_alphas)
POW_SPEC = ElementwiseSpec("pow", compute_pow_relaxation, _pow_alphas, extract_extra=_pow_extra)


# clamp's compute fn takes positional ``min_val``/``max_val`` between the input bounds and α kwargs.
# Wrap to keep the spec's keyword-only convention for ``compute``.
def _compute_clamp(
    input_bounds: IntervalBounds,
    *,
    min_val: float | torch.Tensor | None,
    max_val: float | torch.Tensor | None,
    alpha_clamp_crosses_min_lower: torch.Tensor | None = None,
    alpha_clamp_crosses_max_upper: torch.Tensor | None = None,
) -> ElementwiseParams:
    return compute_clamp_relaxation(  # ty:ignore[no-matching-overload]
        input_bounds,
        min_val,
        max_val,
        alpha_clamp_crosses_min_lower=alpha_clamp_crosses_min_lower,
        alpha_clamp_crosses_max_upper=alpha_clamp_crosses_max_upper,
    )


CLAMP_SPEC = ElementwiseSpec("clamp", _compute_clamp, _clamp_alphas, extract_extra=_clamp_extra)
