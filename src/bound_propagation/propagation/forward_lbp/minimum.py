from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPMinimum(ForwardLBPStrategy):
    """Forward LBP strategy for element-wise minimum (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            lower_a, upper_a = left.concretize()
            lower_b, upper_b = right.concretize()
            region = left.region
        elif isinstance(left, LinearBounds):
            lower_a, upper_a = left.concretize()
            right_tensor = torch.as_tensor(right, dtype=lower_a.dtype, device=lower_a.device)
            lower_b, upper_b = right_tensor, right_tensor
            region = left.region
        elif isinstance(right, LinearBounds):
            left_tensor = torch.as_tensor(left, dtype=right.bias_lower.dtype, device=right.bias_lower.device)
            lower_a, upper_a = left_tensor, left_tensor
            lower_b, upper_b = right.concretize()
            region = right.region
        else:
            raise TypeError(f"ForwardLBPMinimum requires at least one LinearBounds, got {type(left)} and {type(right)}")

        return LinearBounds(
            region=region,
            linear_lower=None,
            bias_lower=torch.minimum(lower_a, lower_b),
            linear_upper=None,
            bias_upper=torch.minimum(upper_a, upper_b),
        )
