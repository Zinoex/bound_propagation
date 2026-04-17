from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from .base import SymbolicLinearRelaxation, _merge_backward_bounds


@final
@dataclass
class SymbolicLinear(SymbolicLinearRelaxation):
    """Backward through ``nn.Linear`` / ``F.linear``: ``y = x @ W^T + b``."""

    weight: torch.Tensor  # (out_features, in_features)
    bias: torch.Tensor | None  # (out_features,) or None
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # A: (*batch, *bounded_out, out_features)
        # W: (out_features, in_features) -> A @ W gives (..., in_features)
        new_A_lower = A_lower @ self.weight
        new_A_upper = A_upper @ self.weight

        bounds = self.input.backward(new_A_lower, new_A_upper, batch_ndim)

        if self.bias is not None:
            # delta_bias = A @ bias: (..., out_features) @ (out_features,) -> (...)
            delta_bias_lower = A_lower @ self.bias
            delta_bias_upper = A_upper @ self.bias
            return LinearBounds(
                regions=bounds.regions,
                linear_lower=bounds.linear_lowers,
                bias_lower=bounds.bias_lower + delta_bias_lower,
                linear_upper=bounds.linear_uppers,
                bias_upper=bounds.bias_upper + delta_bias_upper,
                input_ids=bounds.input_ids,
                validate=False,
            )
        return bounds


@final
@dataclass
class SymbolicMatmulRightConstant(SymbolicLinearRelaxation):
    """Backward through ``y = x @ W`` (right operand constant)."""

    weight: torch.Tensor  # (in_features, out_features)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # A: (*batch, *bounded_out, out_features)
        # W.T: (out_features, in_features) -> A @ W.T gives (..., in_features)
        new_A_lower = A_lower @ self.weight.T
        new_A_upper = A_upper @ self.weight.T
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicMatmulLeftConstant(SymbolicLinearRelaxation):
    """Backward through ``y = W @ x`` (left operand constant)."""

    weight: torch.Tensor  # (out_features, in_features)
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        # A: (*batch, *bounded_out, out_features)
        # W: (out_features, in_features) -> A @ W gives (..., in_features)
        new_A_lower = A_lower @ self.weight
        new_A_upper = A_upper @ self.weight
        return self.input.backward(new_A_lower, new_A_upper, batch_ndim)


@final
@dataclass
class SymbolicAddBounds(SymbolicLinearRelaxation):
    """Backward through ``y = x1 + x2`` (both abstract)."""

    input_left: SymbolicLinearRelaxation
    input_right: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        bounds_left = self.input_left.backward(A_lower, A_upper, batch_ndim)
        bounds_right = self.input_right.backward(A_lower, A_upper, batch_ndim)
        zero = torch.zeros_like(bounds_left.bias_lower)
        return _merge_backward_bounds([bounds_left, bounds_right], zero, zero)


@final
@dataclass
class SymbolicSubBounds(SymbolicLinearRelaxation):
    """Backward through ``y = x1 - x2`` (both abstract)."""

    input_left: SymbolicLinearRelaxation
    input_right: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        bounds_left = self.input_left.backward(A_lower, A_upper, batch_ndim)
        bounds_right = self.input_right.backward(-A_lower, -A_upper, batch_ndim)
        zero = torch.zeros_like(bounds_left.bias_lower)
        return _merge_backward_bounds([bounds_left, bounds_right], zero, zero)


@final
@dataclass
class SymbolicConstantAdd(SymbolicLinearRelaxation):
    """Backward through ``y = x + c`` (constant addend)."""

    constant: torch.Tensor
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        bounds = self.input.backward(A_lower, A_upper, batch_ndim)
        node_ndim = self.constant.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - self.constant.ndim

        c_bc = self.constant.reshape(
            self.constant.shape[:batch_ndim] + (1,) * bounded_ndim + self.constant.shape[batch_ndim:]
        )

        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_lower = (A_lower * c_bc).sum(dim=sum_dims) if sum_dims else A_lower * c_bc
        delta_upper = (A_upper * c_bc).sum(dim=sum_dims) if sum_dims else A_upper * c_bc

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=bounds.linear_lowers,
            bias_lower=bounds.bias_lower + delta_lower,
            linear_upper=bounds.linear_uppers,
            bias_upper=bounds.bias_upper + delta_upper,
            input_ids=bounds.input_ids,
            validate=False,
        )


@final
@dataclass
class SymbolicNeg(SymbolicLinearRelaxation):
    """Backward through ``y = -x``."""

    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        return self.input.backward(-A_lower, -A_upper, batch_ndim)


@final
@dataclass
class SymbolicScale(SymbolicLinearRelaxation):
    """Backward through ``y = c * x`` (element-wise constant scale)."""

    scale: torch.Tensor
    input: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        bounded_ndim = A_lower.ndim - self.scale.ndim
        c = self.scale.reshape(self.scale.shape[:batch_ndim] + (1,) * bounded_ndim + self.scale.shape[batch_ndim:])
        return self.input.backward(A_lower * c, A_upper * c, batch_ndim)
