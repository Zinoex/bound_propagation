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

    @staticmethod
    def _matmul_right_constant_linear(
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
                transformed_flat = torch.einsum("...kd,ko->...od", upper_flat, weight_pos) + torch.einsum(
                    "...kd,ko->...od", lower_flat, weight_neg
                )
            else:
                transformed_flat = torch.einsum("...kd,ko->...od", lower_flat, weight_pos) + torch.einsum(
                    "...kd,ko->...od", upper_flat, weight_neg
                )

            transformed.append(transformed_flat.reshape(*batch_shape, weight_pos.shape[1], *input_axes))

        return transformed

    @staticmethod
    def _matmul_left_constant_linear(
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
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            # TODO: Implement this case. It requires handling the interaction of two sets of linear bounds,
            # which is more complex than the constant cases.
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
        if weight.ndim != 2:
            raise ValueError(f"matmul right operand must be 2D, got shape {tuple(weight.shape)}")

        if bounds.bias_lower.shape[-1] != weight.shape[0]:
            raise ValueError(
                "matmul dimension mismatch: "
                f"bounds last dim {bounds.bias_lower.shape[-1]} vs "
                f"weight first dim {weight.shape[0]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim

        linear_lower = self._matmul_right_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=False,
        )

        bias_lower = torch.einsum("...k,ko->...o", bounds.bias_lower, weight_pos) + torch.einsum(
            "...k,ko->...o", bounds.bias_upper, weight_neg
        )

        linear_upper = self._matmul_right_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=True,
        )

        bias_upper = torch.einsum("...k,ko->...o", bounds.bias_upper, weight_pos) + torch.einsum(
            "...k,ko->...o", bounds.bias_lower, weight_neg
        )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )

    def _matmul_left_constant(self, weight: torch.Tensor, bounds: LinearBounds) -> LinearBounds:
        """z = W @ x where x has linear bounds."""
        if weight.ndim != 2:
            raise ValueError(f"matmul left operand must be 2D, got shape {tuple(weight.shape)}")

        if bounds.bias_lower.shape[-1] != weight.shape[1]:
            raise ValueError(
                "matmul dimension mismatch: "
                f"weight second dim {weight.shape[1]} vs "
                f"bounds last dim {bounds.bias_lower.shape[-1]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim

        linear_lower = self._matmul_left_constant_linear(
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

        linear_upper = self._matmul_left_constant_linear(
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

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
