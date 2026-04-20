"""Backward LBP strategies and relaxations for 2D convolution and pooling.

Convolution and average pooling are linear, so their backward pass through an
A-matrix is the adjoint linear map — realised via :func:`F.conv_transpose2d`
with the same hyperparameters (and a depthwise uniform kernel for average
pooling). Max pooling is nonlinear; its relaxation selects the argmax of the
input lower bound as the "winner" position and routes downstream coefficients
there, while accumulating the pool-window slack ``max_upper - max_lower`` into
the bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from beartype.typing import final

from ...bounds import IntervalBounds
from ..linear_relaxations.alpha_resolvers import resolve_maxpool2d_alphas
from .base import BackwardContributions, BackwardLBPStrategy, BackwardRelaxation, IntermediateBoundsProvider

if TYPE_CHECKING:
    from .tape import BackwardTape


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (int(x[0]), int(x[1]))


def _infer_output_padding(
    input_spatial: tuple[int, int],
    output_spatial: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Derive the ``output_padding`` needed for ``conv_transpose2d`` to reach ``input_spatial``.

    ``conv_transpose2d`` output size formula::

        out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1

    Solving for ``output_padding`` given a target ``input_spatial`` (the shape
    of the pre-conv tensor).
    """
    h_in, w_in = input_spatial
    h_out, w_out = output_spatial
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    h_no_op = (h_out - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1
    w_no_op = (w_out - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1

    op_h = h_in - h_no_op
    op_w = w_in - w_no_op
    if op_h < 0 or op_w < 0 or op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"Cannot infer conv_transpose2d output_padding: "
            f"output_padding=({op_h}, {op_w}) for input_spatial={input_spatial}, "
            f"output_spatial={output_spatial}, stride={stride}"
        )
    return op_h, op_w


def _conv2d_adjoint(
    A: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    input_spatial_shape: tuple[int, int],
) -> torch.Tensor:
    """Apply ``F.conv2d`` adjoint to ``A``.

    ``A`` has shape ``(*leading, C_out, H_out, W_out)`` and the result has shape
    ``(*leading, C_in, H_in, W_in)``. Leading dims (batch + bounded-out) are
    folded into a single batch for :func:`F.conv_transpose2d`.
    """
    C_out, h_out, w_out = A.shape[-3], A.shape[-2], A.shape[-1]
    leading = A.shape[:-3]
    A_flat = A.reshape(-1, C_out, h_out, w_out)

    k_h, k_w = weight.shape[-2], weight.shape[-1]
    output_padding = _infer_output_padding(
        input_spatial=input_spatial_shape,
        output_spatial=(h_out, w_out),
        kernel_size=(k_h, k_w),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    out_flat = F.conv_transpose2d(
        A_flat,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_padding=output_padding,
    )
    C_in = out_flat.shape[-3]
    h_in, w_in = input_spatial_shape
    return out_flat.reshape(*leading, C_in, h_in, w_in)


def _zero_bias(A: torch.Tensor, node_ndim: int) -> torch.Tensor:
    bias_shape = A.shape[: A.ndim - node_ndim] if node_ndim > 0 else A.shape
    return torch.zeros(bias_shape, dtype=A.dtype, device=A.device)


# ----------------------------------------------------------------------
# Conv2d
# ----------------------------------------------------------------------


class BackwardLBPConv2d(BackwardLBPStrategy):
    """Backward LBP strategy for ``nn.Conv2d`` / ``F.conv2d``."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]

        if node.op == "call_module":
            module = tape.get_module(node.target)
            if not isinstance(module, nn.Conv2d):
                raise TypeError(f"BackwardLBPConv2d expected nn.Conv2d, got {type(module).__name__}")
            weight = module.weight
            bias = module.bias
            conv_kwargs: dict[str, Any] = {
                "stride": _pair(module.stride),
                "padding": _pair(module.padding),
                "dilation": _pair(module.dilation),
                "groups": module.groups,
            }
        else:
            weight = args[1] if len(args) > 1 else kwargs["weight"]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            conv_kwargs = {
                "stride": _pair(args[3] if len(args) > 3 else kwargs.get("stride", 1)),
                "padding": _pair(args[4] if len(args) > 4 else kwargs.get("padding", 0)),
                "dilation": _pair(args[5] if len(args) > 5 else kwargs.get("dilation", 1)),
                "groups": int(args[6] if len(args) > 6 else kwargs.get("groups", 1)),
            }

        if weight.ndim != 4:
            raise ValueError(f"conv2d weight must be 4D, got shape {tuple(weight.shape)}")

        input_bounds = bounds(input_node)
        input_shape = tuple(input_bounds.lower.shape)
        output_shape = tuple(node.meta["tensor_meta"]["shape"])

        return Conv2dBackwardRelaxation(
            weight=weight,
            bias=bias,
            stride=conv_kwargs["stride"],
            padding=conv_kwargs["padding"],
            dilation=conv_kwargs["dilation"],
            groups=conv_kwargs["groups"],
            input_shape=input_shape,
            output_ndim=len(output_shape),
            input_node=input_node,
        )


@final
@dataclass
class Conv2dBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = conv2d(x, W) + b``.

    Parameters
    ----------
    weight : torch.Tensor
        Convolution kernel of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : torch.Tensor | None
        Per-output-channel bias of shape ``(C_out,)``, or ``None``.
    stride, padding, dilation : tuple[int, int]
        Conv hyperparameters.
    groups : int
        Number of blocked connections from input to output channels.
    input_shape : tuple[int, ...]
        Shape of the conv input tensor, ``(*batch, C_in, H_in, W_in)``. Used to
        recover ``output_padding`` for the adjoint conv_transpose.
    input_node : fx.Node
        The fx graph node for the input.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_shape: tuple[int, ...]
    output_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        h_in, w_in = int(self.input_shape[-2]), int(self.input_shape[-1])
        new_A_lower = _conv2d_adjoint(
            A_lower, self.weight, self.stride, self.padding, self.dilation, self.groups, (h_in, w_in)
        )
        new_A_upper = _conv2d_adjoint(
            A_upper, self.weight, self.stride, self.padding, self.dilation, self.groups, (h_in, w_in)
        )

        # Node ndim after peeling tape batch: includes any user-level batch dims
        # plus the (C_out, H_out, W_out) conv-output axes.
        node_ndim = self.output_ndim - batch_ndim
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()

        if self.bias is not None:
            # bias shape: (C_out,). Broadcast to match A's trailing node axes and
            # sum over the full node shape so the bias lives on `*batch, *bounded_out`.
            bias_bc = self.bias.view(*([1] * (A_lower.ndim - 3)), -1, 1, 1)
            bias_lower = (A_lower * bias_bc).sum(dim=sum_dims) if sum_dims else A_lower * bias_bc
            bias_upper = (A_upper * bias_bc).sum(dim=sum_dims) if sum_dims else A_upper * bias_bc
        else:
            zero = _zero_bias(A_lower, node_ndim=node_ndim)
            bias_lower = zero
            bias_upper = zero

        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=bias_lower,
            bias_upper=bias_upper,
        )


# ----------------------------------------------------------------------
# AvgPool2d
# ----------------------------------------------------------------------


class BackwardLBPAvgPool2d(BackwardLBPStrategy):
    """Backward LBP strategy for ``nn.AvgPool2d`` / ``F.avg_pool2d``.

    ``nn.AdaptiveAvgPool2d`` and ``F.adaptive_avg_pool2d`` are also routed here;
    the relaxation converts the adaptive operation to an equivalent
    fixed-kernel avg-pool at build time.
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]

        input_bounds = bounds(input_node)
        input_shape = tuple(input_bounds.lower.shape)

        if node.op == "call_module":
            module = tape.get_module(node.target)
            if isinstance(module, nn.AdaptiveAvgPool2d):
                pool_kwargs = _adaptive_to_fixed_avgpool(input_shape, module.output_size)
            elif isinstance(module, nn.AvgPool2d):
                pool_kwargs = {
                    "kernel_size": _pair(module.kernel_size),
                    "stride": _pair(module.stride if module.stride is not None else module.kernel_size),
                    "padding": _pair(module.padding),
                    "ceil_mode": module.ceil_mode,
                    "count_include_pad": module.count_include_pad,
                    "divisor_override": module.divisor_override,
                }
            else:
                raise TypeError(f"BackwardLBPAvgPool2d got unexpected module type {type(module).__name__}")
        elif node.target is F.adaptive_avg_pool2d:
            output_size = args[1] if len(args) > 1 else kwargs["output_size"]
            pool_kwargs = _adaptive_to_fixed_avgpool(input_shape, output_size)
        else:
            kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
            stride = args[2] if len(args) > 2 else kwargs.get("stride")
            if stride is None:
                stride = kernel_size
            pool_kwargs = {
                "kernel_size": _pair(kernel_size),
                "stride": _pair(stride),
                "padding": _pair(args[3] if len(args) > 3 else kwargs.get("padding", 0)),
                "ceil_mode": bool(args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)),
                "count_include_pad": bool(args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)),
                "divisor_override": args[6] if len(args) > 6 else kwargs.get("divisor_override"),
            }

        if pool_kwargs["ceil_mode"]:
            raise NotImplementedError("BackwardLBPAvgPool2d does not support ceil_mode=True")
        if not pool_kwargs["count_include_pad"] and pool_kwargs["padding"] != (0, 0):
            raise NotImplementedError(
                "BackwardLBPAvgPool2d does not support count_include_pad=False with non-zero padding"
            )

        output_shape = tuple(node.meta["tensor_meta"]["shape"])

        return AvgPool2dBackwardRelaxation(
            kernel_size=pool_kwargs["kernel_size"],
            stride=pool_kwargs["stride"],
            padding=pool_kwargs["padding"],
            divisor_override=pool_kwargs["divisor_override"],
            input_shape=input_shape,
            output_ndim=len(output_shape),
            input_node=input_node,
        )


def _adaptive_to_fixed_avgpool(
    input_shape: tuple[int, ...], output_size: int | tuple[int, int] | tuple[int | None, int | None]
) -> dict[str, Any]:
    """Convert ``F.adaptive_avg_pool2d`` args into ``F.avg_pool2d`` equivalents.

    The equivalence only holds when ``H_in`` / ``H_out`` and ``W_in`` / ``W_out``
    divide evenly. Otherwise the adaptive op uses per-cell window sizes and we
    bail out.
    """
    h_in, w_in = int(input_shape[-2]), int(input_shape[-1])
    if isinstance(output_size, int):
        h_out, w_out = output_size, output_size
    else:
        h_out = h_in if output_size[0] is None else int(output_size[0])
        w_out = w_in if output_size[1] is None else int(output_size[1])

    if h_in % h_out != 0 or w_in % w_out != 0:
        raise NotImplementedError(
            "BackwardLBPAvgPool2d for adaptive_avg_pool2d requires input spatial dims divisible by output_size; "
            f"got input={(h_in, w_in)}, output={(h_out, w_out)}"
        )
    k_h = h_in // h_out
    k_w = w_in // w_out
    return {
        "kernel_size": (k_h, k_w),
        "stride": (k_h, k_w),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }


@final
@dataclass
class AvgPool2dBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for average pooling.

    Adjoint is a depthwise transposed convolution with a uniform kernel of
    magnitude ``1 / (kH * kW)`` (or ``1 / divisor_override``).
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    divisor_override: int | None
    input_shape: tuple[int, ...]
    output_ndim: int
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        C = A_lower.shape[-3]
        k_h, k_w = self.kernel_size
        divisor = float(k_h * k_w if self.divisor_override is None else self.divisor_override)
        kernel = torch.full(
            (C, 1, k_h, k_w),
            1.0 / divisor,
            dtype=A_lower.dtype,
            device=A_lower.device,
        )
        h_in, w_in = int(self.input_shape[-2]), int(self.input_shape[-1])
        new_A_lower = _conv2d_adjoint(
            A_lower, kernel, self.stride, self.padding, (1, 1), groups=C, input_spatial_shape=(h_in, w_in)
        )
        new_A_upper = _conv2d_adjoint(
            A_upper, kernel, self.stride, self.padding, (1, 1), groups=C, input_spatial_shape=(h_in, w_in)
        )

        node_ndim = self.output_ndim - batch_ndim
        zero = _zero_bias(A_lower, node_ndim=node_ndim)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )


# ----------------------------------------------------------------------
# MaxPool2d
# ----------------------------------------------------------------------


class BackwardLBPMaxPool2d(BackwardLBPStrategy):
    """Backward LBP strategy for ``nn.MaxPool2d`` / ``F.max_pool2d``.

    The relaxation routes downstream coefficients to the argmax position of
    the input lower bound within each pool window and accumulates the slack
    ``max_upper - max_lower`` into the upper-bound bias.
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]

        input_bounds = bounds(input_node)
        input_shape = tuple(input_bounds.lower.shape)

        if node.op == "call_module":
            module = tape.get_module(node.target)
            if isinstance(module, nn.AdaptiveMaxPool2d):
                pool_kwargs = _adaptive_to_fixed_maxpool(input_shape, module.output_size)
            elif isinstance(module, nn.MaxPool2d):
                pool_kwargs = {
                    "kernel_size": _pair(module.kernel_size),
                    "stride": _pair(module.stride if module.stride is not None else module.kernel_size),
                    "padding": _pair(module.padding),
                    "dilation": _pair(module.dilation),
                    "ceil_mode": module.ceil_mode,
                }
            else:
                raise TypeError(f"BackwardLBPMaxPool2d got unexpected module type {type(module).__name__}")
        elif node.target is F.adaptive_max_pool2d:
            output_size = args[1] if len(args) > 1 else kwargs["output_size"]
            pool_kwargs = _adaptive_to_fixed_maxpool(input_shape, output_size)
        else:
            kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
            stride = args[2] if len(args) > 2 else kwargs.get("stride")
            if stride is None:
                stride = kernel_size
            pool_kwargs = {
                "kernel_size": _pair(kernel_size),
                "stride": _pair(stride),
                "padding": _pair(args[3] if len(args) > 3 else kwargs.get("padding", 0)),
                "dilation": _pair(args[4] if len(args) > 4 else kwargs.get("dilation", 1)),
                "ceil_mode": bool(args[5] if len(args) > 5 else kwargs.get("ceil_mode", False)),
            }

        if pool_kwargs["ceil_mode"]:
            raise NotImplementedError("BackwardLBPMaxPool2d does not support ceil_mode=True")

        output_shape = tuple(node.meta["tensor_meta"]["shape"])

        # Peek at the pool output spatial shape via a zero-cost dry run on the
        # lower bound. Used to size the alpha override.
        with torch.no_grad():
            pool_out_shape = F.max_pool2d(
                input_bounds.lower,
                pool_kwargs["kernel_size"],
                pool_kwargs["stride"],
                pool_kwargs["padding"],
                pool_kwargs["dilation"],
                ceil_mode=False,
            ).shape
        alpha_lower, alpha_upper = resolve_maxpool2d_alphas(
            tape.alpha_provider,
            node,
            output_shape=pool_out_shape,
            device=input_bounds.lower.device,
            dtype=input_bounds.lower.dtype,
        )

        return MaxPool2dBackwardRelaxation(
            kernel_size=pool_kwargs["kernel_size"],
            stride=pool_kwargs["stride"],
            padding=pool_kwargs["padding"],
            dilation=pool_kwargs["dilation"],
            input_bounds=input_bounds,
            input_shape=input_shape,
            output_ndim=len(output_shape),
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            input_node=input_node,
        )


def _adaptive_to_fixed_maxpool(
    input_shape: tuple[int, ...], output_size: int | tuple[int, int] | tuple[int | None, int | None]
) -> dict[str, Any]:
    h_in, w_in = int(input_shape[-2]), int(input_shape[-1])
    if isinstance(output_size, int):
        h_out, w_out = output_size, output_size
    else:
        h_out = h_in if output_size[0] is None else int(output_size[0])
        w_out = w_in if output_size[1] is None else int(output_size[1])

    if h_in % h_out != 0 or w_in % w_out != 0:
        raise NotImplementedError(
            "BackwardLBPMaxPool2d for adaptive_max_pool2d requires input spatial dims divisible by output_size; "
            f"got input={(h_in, w_in)}, output={(h_out, w_out)}"
        )
    k_h = h_in // h_out
    k_w = w_in // w_out
    return {
        "kernel_size": (k_h, k_w),
        "stride": (k_h, k_w),
        "padding": (0, 0),
        "dilation": (1, 1),
        "ceil_mode": False,
    }


@final
@dataclass
class MaxPool2dBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for 2D max pooling with argmax-of-lower winner routing.

    For each output cell ``i* = argmax_{i in window} lower[i]``, the relaxation
    interpolates between winner-routing and the IBP constant fallback via two
    optional alpha-CROWN knobs in ``[0, 1]``::

        y_lower ≥ alpha_l · x[i*] + (1 − alpha_l) · max_lower
        y_upper ≤ alpha_u · x[i*] + max_upper − alpha_u · max_lower

    Both are sound convex combinations of two valid bounds for any
    ``alpha ∈ [0, 1]``. At ``alpha = 1`` they reduce to pure winner-routing
    (``y_lower ≥ x[i*]`` and ``y_upper ≤ x[i*] + max_upper − max_lower``); at
    ``alpha = 0`` they reduce to IBP constants. When ``alpha_lower`` /
    ``alpha_upper`` are ``None`` (alpha optimization disabled for this node),
    the analytical default ``alpha = 1`` is used and the fast path skips the
    per-sign split of the routed A.
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    input_bounds: IntervalBounds
    input_shape: tuple[int, ...]
    output_ndim: int
    alpha_lower: torch.Tensor | None
    alpha_upper: torch.Tensor | None
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        lower_in = self.input_bounds.lower
        upper_in = self.input_bounds.upper

        max_lower, indices = F.max_pool2d(
            lower_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=False,
            return_indices=True,
        )
        max_upper = F.max_pool2d(
            upper_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=False,
        )

        node_ndim = self.output_ndim - batch_ndim
        bounded_ndim = A_lower.ndim - max_lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        if self.alpha_lower is None and self.alpha_upper is None:
            # Fast path (no alpha optimization): slope is 1 at i* everywhere.
            new_A_lower = _route_A_via_indices(A_lower, indices, self.input_shape, batch_ndim, node_ndim)
            new_A_upper = _route_A_via_indices(A_upper, indices, self.input_shape, batch_ndim, node_ndim)

            slack_bc = bc((max_upper - max_lower).clamp(min=0))
            A_l_neg = A_lower.clamp(max=0)
            A_u_pos = A_upper.clamp(min=0)
            sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
            if sum_dims:
                delta_lower = (A_l_neg * slack_bc).sum(dim=sum_dims)
                delta_upper = (A_u_pos * slack_bc).sum(dim=sum_dims)
            else:
                delta_lower = A_l_neg * slack_bc
                delta_upper = A_u_pos * slack_bc

            return BackwardContributions(
                a_terms={self.input_node: (new_A_lower, new_A_upper)},
                bias_lower=delta_lower,
                bias_upper=delta_upper,
            )

        # Alpha-CROWN path: interpolate winner-routing vs. IBP constants.
        alpha_l = self.alpha_lower if self.alpha_lower is not None else torch.ones_like(max_lower)
        alpha_u = self.alpha_upper if self.alpha_upper is not None else torch.ones_like(max_lower)
        alpha_l_bc = bc(alpha_l)
        alpha_u_bc = bc(alpha_u)

        # Sign-decomposed slope selection. For the lower A:
        #   A_l_pos uses the lower relaxation's slope (alpha_l at i*)
        #   A_l_neg uses the upper relaxation's slope (alpha_u at i*)
        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        A_lower_eff = A_l_pos * alpha_l_bc + A_l_neg * alpha_u_bc
        A_upper_eff = A_u_pos * alpha_u_bc + A_u_neg * alpha_l_bc

        new_A_lower = _route_A_via_indices(A_lower_eff, indices, self.input_shape, batch_ndim, node_ndim)
        new_A_upper = _route_A_via_indices(A_upper_eff, indices, self.input_shape, batch_ndim, node_ndim)

        # Bias contributions.
        #   bias_lower_relax = (1 − alpha_l) · max_lower
        #   bias_upper_relax = max_upper − alpha_u · max_lower
        bias_lower_relax = (1.0 - alpha_l) * max_lower
        bias_upper_relax = max_upper - alpha_u * max_lower
        bias_lower_relax_bc = bc(bias_lower_relax)
        bias_upper_relax_bc = bc(bias_upper_relax)

        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_lower = A_l_pos * bias_lower_relax_bc + A_l_neg * bias_upper_relax_bc
        delta_upper = A_u_pos * bias_upper_relax_bc + A_u_neg * bias_lower_relax_bc
        if sum_dims:
            delta_lower = delta_lower.sum(dim=sum_dims)
            delta_upper = delta_upper.sum(dim=sum_dims)

        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=delta_lower,
            bias_upper=delta_upper,
        )


def _route_A_via_indices(
    A: torch.Tensor,
    indices: torch.Tensor,
    input_shape: tuple[int, ...],
    batch_ndim: int,
    node_ndim: int,
) -> torch.Tensor:
    """Scatter ``A`` along spatial axes according to ``indices``.

    Parameters
    ----------
    A : torch.Tensor
        Coefficient tensor of shape ``(*batch, *bounded_out, *node)`` where the
        trailing ``node`` axes end with ``(C, H_out, W_out)``; any user batch
        dims in between are absorbed into ``node`` when ``batch_ndim`` < rank.
    indices : torch.Tensor
        Max-pool argmax indices of shape ``(*batch, *node)`` with values in
        ``[0, H_in * W_in)``.
    input_shape : tuple[int, ...]
        Shape of the pool input tensor, ``(*batch, *node_in)``.
    batch_ndim, node_ndim : int
        Tape batch rank and the full node rank (output_ndim - batch_ndim).
    """
    h_out, w_out = A.shape[-2], A.shape[-1]
    h_in, w_in = int(input_shape[-2]), int(input_shape[-1])

    bounded_ndim = A.ndim - batch_ndim - node_ndim

    # Broadcast A and indices to a common shape along the tape batch axis.
    # A typically carries size-1 along the tape batch (broadcast identity),
    # while indices carries the actual batch size from the concrete region.
    idx_shape = indices.shape[:batch_ndim] + (1,) * bounded_ndim + indices.shape[batch_ndim:]
    idx_bc = indices.reshape(idx_shape)
    common_shape = torch.broadcast_shapes(A.shape, idx_bc.shape)
    A = A.expand(common_shape)
    idx_bc = idx_bc.expand(common_shape)

    A_flat = A.reshape(*A.shape[:-2], h_out * w_out)
    idx_flat = idx_bc.reshape(*idx_bc.shape[:-2], h_out * w_out)

    new_A_flat = torch.zeros(*A.shape[:-2], h_in * w_in, dtype=A.dtype, device=A.device)
    new_A_flat.scatter_add_(-1, idx_flat, A_flat)
    return new_A_flat.reshape(*A.shape[:-2], h_in, w_in)


__all__ = [
    "AvgPool2dBackwardRelaxation",
    "BackwardLBPAvgPool2d",
    "BackwardLBPConv2d",
    "BackwardLBPMaxPool2d",
    "Conv2dBackwardRelaxation",
    "MaxPool2dBackwardRelaxation",
]
