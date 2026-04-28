from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from ...errors import DimensionMismatchError
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


def _normalize_matmul_operands(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
    """Normalize operands to matrix form while preserving torch.matmul semantics."""
    if left.ndim == 0 or right.ndim == 0:
        raise DimensionMismatchError(
            "matmul requires tensor inputs with at least 1 dimension, "
            f"got left.ndim={left.ndim} and right.ndim={right.ndim}"
        )

    squeeze_left = left.ndim == 1
    squeeze_right = right.ndim == 1

    if squeeze_left:
        left = left.unsqueeze(0)
    if squeeze_right:
        right = right.unsqueeze(-1)

    if left.shape[-1] != right.shape[-2]:
        raise DimensionMismatchError(
            "matmul requires compatible reduction dimensions, "
            f"got left.shape={tuple(left.shape)} and right.shape={tuple(right.shape)}"
        )

    try:
        batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    except RuntimeError as error:
        raise DimensionMismatchError(
            "matmul requires broadcastable batch dimensions, "
            f"got left.shape={tuple(left.shape)} and right.shape={tuple(right.shape)}"
        ) from error

    left = left.expand(*batch_shape, left.shape[-2], left.shape[-1])
    right = right.expand(*batch_shape, right.shape[-2], right.shape[-1])

    return left, right, squeeze_left, squeeze_right


def _restore_matmul_output(result: torch.Tensor, squeeze_left: bool, squeeze_right: bool) -> torch.Tensor:
    """Restore vector/scalar outputs after matrix-form computation."""
    if squeeze_left:
        result = result.squeeze(-2)
    if squeeze_right:
        result = result.squeeze(-1)
    return result


def _matmul_interval_with_constant(interval: IntervalBounds, constant: torch.Tensor) -> IntervalBounds:
    """Compute exact interval bounds for interval @ constant."""
    lower, constant_normalized, squeeze_left, squeeze_right = _normalize_matmul_operands(
        interval.lower,
        constant,
    )
    upper, _, _, _ = _normalize_matmul_operands(interval.upper, constant)

    constant_positive = torch.clamp(constant_normalized, min=0)
    constant_negative = torch.clamp(constant_normalized, max=0)

    lower_result = torch.matmul(lower, constant_positive) + torch.matmul(upper, constant_negative)
    upper_result = torch.matmul(upper, constant_positive) + torch.matmul(lower, constant_negative)

    return IntervalBounds(
        _restore_matmul_output(lower_result, squeeze_left, squeeze_right),
        _restore_matmul_output(upper_result, squeeze_left, squeeze_right),
    )


def _matmul_constant_with_interval(constant: torch.Tensor, interval: IntervalBounds) -> IntervalBounds:
    """Compute exact interval bounds for constant @ interval."""
    constant_normalized, lower, squeeze_left, squeeze_right = _normalize_matmul_operands(
        constant,
        interval.lower,
    )
    _, upper, _, _ = _normalize_matmul_operands(constant, interval.upper)

    constant_positive = torch.clamp(constant_normalized, min=0)
    constant_negative = torch.clamp(constant_normalized, max=0)

    lower_result = torch.matmul(constant_positive, lower) + torch.matmul(constant_negative, upper)
    upper_result = torch.matmul(constant_positive, upper) + torch.matmul(constant_negative, lower)

    return IntervalBounds(
        _restore_matmul_output(lower_result, squeeze_left, squeeze_right),
        _restore_matmul_output(upper_result, squeeze_left, squeeze_right),
    )


def _matmul_interval_with_interval(left: IntervalBounds, right: IntervalBounds) -> IntervalBounds:
    """Compute exact interval bounds for interval @ interval."""
    left_lower, right_lower, squeeze_left, squeeze_right = _normalize_matmul_operands(
        left.lower,
        right.lower,
    )
    left_upper, _, _, _ = _normalize_matmul_operands(left.upper, right.lower)
    _, right_upper, _, _ = _normalize_matmul_operands(left.lower, right.upper)

    ll = left_lower.unsqueeze(-1) * right_lower.unsqueeze(-3)
    lu = left_lower.unsqueeze(-1) * right_upper.unsqueeze(-3)
    ul = left_upper.unsqueeze(-1) * right_lower.unsqueeze(-3)
    uu = left_upper.unsqueeze(-1) * right_upper.unsqueeze(-3)

    lower_result = torch.minimum(torch.minimum(ll, lu), torch.minimum(ul, uu)).sum(dim=-2)
    upper_result = torch.maximum(torch.maximum(ll, lu), torch.maximum(ul, uu)).sum(dim=-2)

    return IntervalBounds(
        _restore_matmul_output(lower_result, squeeze_left, squeeze_right),
        _restore_matmul_output(upper_result, squeeze_left, squeeze_right),
    )


class IBPMatmul(ForwardIBPStrategy):
    """IBP strategy for matmul (all combinations of abstract/constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return _matmul_interval_with_interval(left, right)

        if isinstance(left, IntervalBounds) and isinstance(right, torch.Tensor):
            return _matmul_interval_with_constant(left, right)

        if isinstance(left, torch.Tensor) and isinstance(right, IntervalBounds):
            return _matmul_constant_with_interval(left, right)

        raise TypeError(f"IBPMatmul requires at least one IntervalBounds, got {type(left)} and {type(right)}")
