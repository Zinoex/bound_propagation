from __future__ import annotations

import torch

from .base import ElementwiseLinearRelaxation
from .reciprocal import compute_reciprocal_relaxation


def compute_constant_div_relaxation(
    lower: torch.Tensor,
    upper: torch.Tensor,
    constant: object,
    zero_threshold: float = 1e-8,
) -> ElementwiseLinearRelaxation:
    """Compute linear-relaxation parameters for constant-over-bounds division ``constant / x``.

    The function ``f(x) = c / x`` is a scaled reciprocal.

    - For ``c > 0``, lower/upper relaxations keep the same orientation as reciprocal.
    - For ``c < 0``, multiplying by a negative constant flips inequalities, so lower/upper are swapped.
    - For intervals crossing zero, returns ``(-inf, inf)`` bounds element-wise.

    Args:
        lower: Lower bounds of denominator ``x``.
        upper: Upper bounds of denominator ``x``.
        constant: Numerator constant (scalar or tensor broadcastable to ``lower``).
        zero_threshold: Threshold used by reciprocal relaxation for zero-width handling.

    Returns:
        ElementwiseLinearRelaxation encapsulating the relaxation.
    """
    if not isinstance(lower, torch.Tensor):
        raise TypeError(f"lower must be a torch.Tensor, got {type(lower)!r}")
    if not isinstance(upper, torch.Tensor):
        raise TypeError(f"upper must be a torch.Tensor, got {type(upper)!r}")
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must have the same shape, got {lower.shape} and {upper.shape}")
    if zero_threshold < 0:
        raise ValueError(f"zero_threshold must be non-negative, got {zero_threshold}")

    constant_tensor = torch.as_tensor(constant, dtype=lower.dtype, device=lower.device)
    try:
        constant_tensor = torch.broadcast_to(constant_tensor, lower.shape)
    except RuntimeError as error:
        constant_shape = tuple(constant_tensor.shape)
        raise ValueError(
            f"constant must be broadcastable to denominator bounds shape {lower.shape}, "
            f"got constant shape {constant_shape}"
        ) from error

    recip = compute_reciprocal_relaxation(lower, upper, zero_threshold=zero_threshold)

    positive_constant = constant_tensor > 0
    zero_constant = constant_tensor == 0
    crosses_zero = (lower < 0) & (upper > 0)

    # Multiplication by a negative constant flips lower/upper inequalities.
    alpha_lower = constant_tensor * torch.where(positive_constant, recip.alpha_lower, recip.alpha_upper)
    beta_lower = constant_tensor * torch.where(positive_constant, recip.beta_lower, recip.beta_upper)
    alpha_upper = constant_tensor * torch.where(positive_constant, recip.alpha_upper, recip.alpha_lower)
    beta_upper = constant_tensor * torch.where(positive_constant, recip.beta_upper, recip.beta_lower)

    # For intervals crossing zero, the function contains asymptotes, so return infinite bounds.
    alpha_lower[crosses_zero] = 0.0
    beta_lower[crosses_zero] = float("-inf")
    alpha_upper[crosses_zero] = 0.0
    beta_upper[crosses_zero] = float("inf")

    # For zero constant, the function is constant zero, so alpha=0, beta=0.
    # This must take precedence over zero-crossing handling.
    alpha_lower[zero_constant] = 0.0
    beta_lower[zero_constant] = 0.0
    alpha_upper[zero_constant] = 0.0
    beta_upper[zero_constant] = 0.0

    return ElementwiseLinearRelaxation(
        alpha_lower=alpha_lower,
        beta_lower=beta_lower,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
