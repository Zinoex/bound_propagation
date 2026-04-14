from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPMatmul(ForwardLBPStrategy):
    """Forward LBP strategy for matmul (abstract@abstract, abstract@constant, constant@abstract)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            raise NotImplementedError(
                "LBP matmul with two varying operands not yet supported. Use constant weights or switch to IBP method."
            )

        if isinstance(left, LinearBounds) and isinstance(right, torch.Tensor):
            return self._matmul_right_constant(left, right)

        if isinstance(left, torch.Tensor) and isinstance(right, LinearBounds):
            return self._matmul_left_constant(left, right)

        raise TypeError(f"ForwardLBPMatmul requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _matmul_right_constant(self, bounds: LinearBounds, weight: torch.Tensor) -> LinearBounds:
        """z = x @ W where x has linear bounds."""
        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower = bounds.linear_lower @ weight_pos + bounds.linear_upper @ weight_neg
        elif bounds.linear_lower is not None:
            linear_lower = bounds.linear_lower @ weight_pos + bounds.linear_lower @ weight_neg
        elif bounds.linear_upper is not None:
            linear_lower = bounds.linear_upper @ weight_pos + bounds.linear_upper @ weight_neg
        else:
            linear_lower = None

        bias_lower = bounds.bias_lower @ weight_pos + bounds.bias_upper @ weight_neg

        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_upper = bounds.linear_upper @ weight_pos + bounds.linear_lower @ weight_neg
        elif bounds.linear_upper is not None:
            linear_upper = bounds.linear_upper @ weight_pos + bounds.linear_upper @ weight_neg
        elif bounds.linear_lower is not None:
            linear_upper = bounds.linear_lower @ weight_pos + bounds.linear_lower @ weight_neg
        else:
            linear_upper = None

        bias_upper = bounds.bias_upper @ weight_pos + bounds.bias_lower @ weight_neg

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )

    def _matmul_left_constant(self, weight: torch.Tensor, bounds: LinearBounds) -> LinearBounds:
        """z = W @ x where x has linear bounds."""
        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_lower = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        else:
            linear_lower = None

        bias_lower = weight_pos @ bounds.bias_lower + weight_neg @ bounds.bias_upper

        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_upper = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        else:
            linear_upper = None

        bias_upper = weight_pos @ bounds.bias_upper + weight_neg @ bounds.bias_lower

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
