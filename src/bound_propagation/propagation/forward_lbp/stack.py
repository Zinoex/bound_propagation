from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPStack(ForwardLBPStrategy):
    """Forward LBP strategy for torch.stack."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        bounds_list: list[LinearBounds] = []
        for i, t in enumerate(tensors):
            if not isinstance(t, LinearBounds):
                raise TypeError(f"ForwardLBPStack requires all inputs to be LinearBounds, but input {i} is {type(t)}")
            bounds_list.append(t)

        lowers = [b.concretize()[0] for b in bounds_list]
        uppers = [b.concretize()[1] for b in bounds_list]

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=torch.stack(lowers, dim=dim),
            linear_upper=[],
            bias_upper=torch.stack(uppers, dim=dim),
        )
