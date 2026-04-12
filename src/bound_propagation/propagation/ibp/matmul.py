from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


def _normalize_matmul_operands(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
    """Normalize operands to matrix form while preserving torch.matmul semantics."""
    if left.ndim == 0 or right.ndim == 0:
        raise ValueError(
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
        raise ValueError(
            "matmul requires compatible reduction dimensions, "
            f"got left.shape={tuple(left.shape)} and right.shape={tuple(right.shape)}"
        )

    try:
        batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    except RuntimeError as error:
        raise ValueError(
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
    """IBP strategy for MATMUL operation: Z = torch.matmul(X, Y)."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"matmul requires 2 inputs, got {len(input_bounds)}")

        A = input_bounds[0]
        B = input_bounds[1]

        if not isinstance(A, IntervalBounds) or not isinstance(B, IntervalBounds):
            raise TypeError(
                "IBPMatmul requires both inputs to be IntervalBounds, but got "
                f"{type(A)} and {type(B)}"
            )

        return _matmul_interval_with_interval(A, B)


class IBPMatmulConstant(ForwardIBPStrategy):
    """IBP strategy for MATMUL Z = torch.matmul(X, Y) when Y is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"matmul requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        y = input_bounds[1]

        if not isinstance(x, IntervalBounds) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "IBPMatmulConstant requires the first input to be IntervalBounds and "
                f"the second input to be torch.Tensor, got {type(x)} and {type(y)}"
            )

        return _matmul_interval_with_constant(x, y)


class IBPConstantMatmul(ForwardIBPStrategy):
    """IBP strategy for MATMUL Z = torch.matmul(X, Y) when X is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"matmul requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        y = input_bounds[1]

        if not isinstance(x, torch.Tensor) or not isinstance(y, IntervalBounds):
            raise TypeError(
                "IBPConstantMatmul requires the first input to be torch.Tensor and "
                f"the second input to be IntervalBounds, got {type(x)} and {type(y)}"
            )

        return _matmul_constant_with_interval(x, y)
