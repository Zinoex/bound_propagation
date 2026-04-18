from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy
from .utils import combine_linear_terms

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
            if not isinstance(node.target, str):
                raise TypeError(f"Expected node.target to be str for call_module, got {type(node.target)}")

            module = ctx.get_module(node.target)
            weight: torch.Tensor = module.weight  # ty:ignore[invalid-assignment]
            bias: torch.Tensor | None = getattr(module, "bias", None)
        else:
            # F.linear(input, weight, bias=None)
            weight: torch.Tensor = args[1] if len(args) > 1 else kwargs.get("weight")  # ty:ignore[invalid-assignment]
            bias: torch.Tensor | None = args[2] if len(args) > 2 else kwargs.get("bias")

        if weight is None:
            raise ValueError("ForwardLBPLinear requires a weight tensor")

        # Pytorch allows the weight to be either 1D or 2D
        # TODO: consider supporting 1D weight (i.e. elementwise multiplication) as a special case without reshaping
        if weight.ndim not in [1, 2]:
            raise ValueError(f"linear weight must be 1D or 2D, got shape {tuple(weight.shape)}")

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


class ForwardLBPNeg(ForwardLBPStrategy):
    """Forward LBP strategy for negation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPNeg requires input to be LinearBounds")

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=[-linear for linear in bounds.linear_uppers],
            bias_lower=-bounds.bias_upper,
            linear_upper=[-linear for linear in bounds.linear_lowers],
            bias_upper=-bounds.bias_lower,
            input_ids=bounds.input_ids,
        )


class ForwardLBPSub(ForwardLBPStrategy):
    """Forward LBP strategy for subtraction (abstract-abstract or abstract-constant or constant-abstract)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._sub_bounds(left, right)

        if isinstance(left, LinearBounds):
            # x - c
            return LinearBounds(
                regions=left.regions,
                linear_lower=left.linear_lowers,
                bias_lower=left.bias_lower - right,
                linear_upper=left.linear_uppers,
                bias_upper=left.bias_upper - right,
                input_ids=left.input_ids,
            )

        if isinstance(right, LinearBounds):
            # c - x: flip signs and bounds
            return LinearBounds(
                regions=right.regions,
                linear_lower=[-linear for linear in right.linear_uppers],
                bias_lower=left - right.bias_upper,
                linear_upper=[-linear for linear in right.linear_lowers],
                bias_upper=left - right.bias_lower,
                input_ids=right.input_ids,
            )

        raise TypeError(f"ForwardLBPSub requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _sub_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        lower_regions, linear_lower, input_ids = combine_linear_terms([(a, "lower", 1.0), (b, "upper", -1.0)])
        upper_regions, linear_upper, upper_input_ids = combine_linear_terms([(a, "upper", 1.0), (b, "lower", -1.0)])

        if input_ids != upper_input_ids:
            raise ValueError(f"Lower and upper input IDs must match, got {input_ids} vs {upper_input_ids}")

        return LinearBounds(
            regions=lower_regions or upper_regions,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower - b.bias_upper,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper - b.bias_lower,
            input_ids=input_ids,
        )


class ForwardLBPAdd(ForwardLBPStrategy):
    """Forward LBP strategy for addition (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._add_bounds(left, right)

        if isinstance(left, LinearBounds):
            return self._add_constant(left, right)

        if isinstance(right, LinearBounds):
            return self._add_constant(right, left)

        raise TypeError(f"ForwardLBPAdd requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _add_bounds(self, a: LinearBounds, b: LinearBounds) -> LinearBounds:
        lower_regions, linear_lower, input_ids = combine_linear_terms([(a, "lower", 1.0), (b, "lower", 1.0)])
        upper_regions, linear_upper, upper_input_ids = combine_linear_terms([(a, "upper", 1.0), (b, "upper", 1.0)])

        if input_ids != upper_input_ids:
            raise ValueError(f"Lower and upper input IDs must match, got {input_ids} vs {upper_input_ids}")

        return LinearBounds(
            regions=lower_regions or upper_regions,
            linear_lower=linear_lower,
            bias_lower=a.bias_lower + b.bias_lower,
            linear_upper=linear_upper,
            bias_upper=a.bias_upper + b.bias_upper,
            input_ids=input_ids,
        )

    def _add_constant(self, bounds: LinearBounds, constant: torch.Tensor | torch.types.Number) -> LinearBounds:
        return LinearBounds(
            regions=bounds.regions,
            linear_lower=bounds.linear_lowers,
            bias_lower=bounds.bias_lower + constant,
            linear_upper=bounds.linear_uppers,
            bias_upper=bounds.bias_upper + constant,
            input_ids=bounds.input_ids,
        )
