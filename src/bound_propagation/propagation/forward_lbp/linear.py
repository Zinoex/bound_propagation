from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPLinear(ForwardLBPStrategy):
    """Forward LBP strategy for nn.Linear / F.linear."""

    @staticmethod
    def _apply_weight_to_linear_terms(
        linear_lowers: list[torch.Tensor],
        linear_uppers: list[torch.Tensor],
        weight_pos: torch.Tensor,
        weight_neg: torch.Tensor,
        output_ndim: int,
        *,
        upper: bool,
    ) -> list[torch.Tensor]:
        transformed: list[torch.Tensor] = []
        for lower_linear, upper_linear in zip(linear_lowers, linear_uppers, strict=True):
            input_axes = lower_linear.shape[output_ndim:]
            batch_shape = lower_linear.shape[: output_ndim - 1]
            feature_dim = lower_linear.shape[output_ndim - 1]

            lower_flat = lower_linear.reshape(*batch_shape, feature_dim, -1)
            upper_flat = upper_linear.reshape(*batch_shape, feature_dim, -1)

            if upper:
                transformed_flat = torch.einsum("ok,...kd->...od", weight_pos, upper_flat) + torch.einsum(
                    "ok,...kd->...od", weight_neg, lower_flat
                )
            else:
                transformed_flat = torch.einsum("ok,...kd->...od", weight_pos, lower_flat) + torch.einsum(
                    "ok,...kd->...od", weight_neg, upper_flat
                )

            transformed.append(transformed_flat.reshape(*batch_shape, weight_pos.shape[0], *input_axes))

        return transformed

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPLinear requires input to be LinearBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            # F.linear(input, weight, bias=None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

        if weight is None:
            raise ValueError("ForwardLBPLinear requires a weight tensor")

        if weight.ndim != 2:
            raise ValueError(f"linear weight must be 2D, got shape {tuple(weight.shape)}")

        if bounds.bias_lower.shape[-1] != weight.shape[1]:
            raise ValueError(
                "linear dimension mismatch: "
                f"bounds last dim {bounds.bias_lower.shape[-1]} vs "
                f"weight second dim {weight.shape[1]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim

        # Lower bound: weight_pos @ lower_coeffs + weight_neg @ upper_coeffs
        linear_lower = self._apply_weight_to_linear_terms(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=False,
        )

        bias_lower = torch.einsum("ok,...k->...o", weight_pos, bounds.bias_lower) + torch.einsum(
            "ok,...k->...o", weight_neg, bounds.bias_upper
        )
        if bias is not None:
            bias_lower = bias_lower + bias

        # Upper bound: weight_pos @ upper_coeffs + weight_neg @ lower_coeffs
        linear_upper = self._apply_weight_to_linear_terms(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=True,
        )

        bias_upper = torch.einsum("ok,...k->...o", weight_pos, bounds.bias_upper) + torch.einsum(
            "ok,...k->...o", weight_neg, bounds.bias_lower
        )
        if bias is not None:
            bias_upper = bias_upper + bias

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
