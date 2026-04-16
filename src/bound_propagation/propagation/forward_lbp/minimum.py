from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.minimum import compute_minimum_relaxation
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
            return self._min_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._min_bounds(left, self._constant_to_bounds(right, left))

        if isinstance(right, LinearBounds):
            return self._min_bounds(self._constant_to_bounds(left, right), right)

        raise TypeError(f"ForwardLBPMinimum requires at least one LinearBounds, got {type(left)} and {type(right)}")

    @staticmethod
    def _constant_to_bounds(constant: object, reference: LinearBounds) -> LinearBounds:
        constant_tensor = torch.as_tensor(
            constant, dtype=reference.bias_lower.dtype, device=reference.bias_lower.device
        )
        constant_tensor = constant_tensor.expand_as(reference.bias_lower)
        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=constant_tensor,
            linear_upper=[],
            bias_upper=constant_tensor,
        )

    @staticmethod
    def _min_bounds(a: LinearBounds, b: LinearBounds) -> LinearBounds:
        lower_a, upper_a = a.concretize()
        lower_b, upper_b = b.concretize()
        relaxation = compute_minimum_relaxation(lower_a, upper_a, lower_b, upper_b)
        return relaxation.forward_compose([a, b])
