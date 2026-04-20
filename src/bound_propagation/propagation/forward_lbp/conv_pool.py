"""Forward-LBP strategies for 2D convolution and pooling.

This module implements forward-mode LBP for conv/pool by **materializing**
each affine coefficient tensor to the familiar ``(*batch, *feature, *input)``
dense layout, then applying ``F.conv2d`` / ``F.avg_pool2d`` to the feature
axes while preserving the trailing input axes via an axis-permutation trick::

    (*batch, C_in, H_in, W_in, *input_shape)
        → move input_shape to front → flatten with batch
        → F.conv2d
        → unflatten → move back
        → (*batch, C_out, H_out, W_out, *input_shape)

This is behavior-complete for CNN verification and is tight by the standard
CROWN sign-decomposition argument. Its memory use matches "dense CROWN" — for
large CNNs, a structured patch-mode operator is a future-work optimization
(deferred from the initial phase-4 scope).

Max pooling is nonlinear; the relaxation mirrors the backward-LBP version:
argmax-of-lower winner routing plus the slack ``max_upper − max_lower`` folded
into the upper-bound bias, with an optional alpha-CROWN interpolation between
winner-routing and IBP constants (two knobs, init ``1.0``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from ...bounds import LinearBounds
from ...linear_operators import DenseOperator, LinearOperator
from ..linear_relaxations.alpha_resolvers import resolve_maxpool2d_alphas
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


# ----------------------------------------------------------------------
# Shape helpers
# ----------------------------------------------------------------------


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (int(x[0]), int(x[1]))


def _apply_feature_op(
    tensor: torch.Tensor,
    output_ndim: int,
    feature_ndim: int,
    op: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply ``op`` to the ``(*feature)`` axes of ``tensor`` while preserving
    leading ``output_ndim - feature_ndim`` batch/bounded axes and trailing
    ``*input_shape`` axes.

    ``op`` receives a 4-D tensor of shape ``(N, C, H, W)`` and returns a 4-D
    tensor of shape ``(N, C', H', W')``. The caller is responsible for making
    ``op`` operate purely on the (C, H, W) axes (any feature-level
    manipulation).
    """
    # tensor: (*batch, *feature, *input_shape)
    # output_ndim = len(batch) + len(feature); feature axes are the last `feature_ndim`
    # of the output slice.
    batch_ndim = output_ndim - feature_ndim
    batch_shape = tensor.shape[:batch_ndim]
    feat_shape = tensor.shape[batch_ndim:output_ndim]
    input_shape = tensor.shape[output_ndim:]

    # Move input axes to front: (*input_shape, *batch, *feature)
    input_axes_src = tuple(range(output_ndim, tensor.ndim))
    permuted = tensor.permute(*input_axes_src, *range(output_ndim))

    # Flatten (*input_shape, *batch) into N; keep (*feature).
    collapsed = permuted.reshape(-1, *feat_shape)

    result = op(collapsed)  # (N, *new_feature) — op yields 4-D (N, C', H', W')
    new_feat_shape = result.shape[1:]

    # Unflatten back and permute input axes to the tail.
    unflat = result.reshape(*input_shape, *batch_shape, *new_feat_shape)
    n_batch_input = len(input_shape) + len(batch_shape)
    # Inverse permutation of (input_src, batch, feat) → (batch, feat, input_src).
    return unflat.permute(
        *range(len(input_shape), n_batch_input),  # batch
        *range(n_batch_input, n_batch_input + len(new_feat_shape)),  # new feature
        *range(len(input_shape)),  # input axes (back to tail)
    )


# ----------------------------------------------------------------------
# Conv2d
# ----------------------------------------------------------------------


class ForwardLBPConv2d(ForwardLBPStrategy):
    """Forward-LBP strategy for ``nn.Conv2d`` / ``F.conv2d``.

    Each affine term is transformed by applying sign-decomposed conv2d to its
    feature ``(C_in, H_in, W_in)`` axes while leaving the trailing input axes
    intact. The bias terms are transformed identically (and the conv's bias
    is added to both ``bias_lower`` and ``bias_upper``).
    """

    def propagate_forward(self, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPConv2d requires input to be LinearBounds")

        weight, bias, conv_kwargs = _resolve_conv2d_params(node, ctx, args, kwargs)
        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        output_ndim = bounds.bias_lower.ndim
        feature_ndim = 3  # (C_in, H_in, W_in)

        def conv_pos(t: torch.Tensor) -> torch.Tensor:
            return F.conv2d(t, weight_pos, bias=None, **conv_kwargs)

        def conv_neg(t: torch.Tensor) -> torch.Tensor:
            return F.conv2d(t, weight_neg, bias=None, **conv_kwargs)

        # Transform each input affine term. new_lower = conv_pos(lower) + conv_neg(upper).
        new_linear_lower: list[LinearOperator] = []
        new_linear_upper: list[LinearOperator] = []
        for lower_op, upper_op in zip(bounds.linear_lowers_op, bounds.linear_uppers_op, strict=True):
            lower_t = lower_op.to_dense().tensor
            upper_t = upper_op.to_dense().tensor

            lp = _apply_feature_op(lower_t, output_ndim, feature_ndim, conv_pos)
            ln = _apply_feature_op(upper_t, output_ndim, feature_ndim, conv_neg)
            up = _apply_feature_op(upper_t, output_ndim, feature_ndim, conv_pos)
            un = _apply_feature_op(lower_t, output_ndim, feature_ndim, conv_neg)

            new_output_shape = (
                lp.shape[: output_ndim - feature_ndim] + lp.shape[output_ndim - feature_ndim : output_ndim]
            )
            new_linear_lower.append(DenseOperator(lp + ln, output_shape=new_output_shape))
            new_linear_upper.append(DenseOperator(up + un, output_shape=new_output_shape))

        # Transform the bias terms the same way (but treating them as 0-input-axes terms).
        def conv_on_bias(t: torch.Tensor, pos: bool) -> torch.Tensor:
            kern = weight_pos if pos else weight_neg
            return F.conv2d(t, kern, bias=None, **conv_kwargs)

        new_bias_lower = _apply_feature_op(
            bounds.bias_lower, output_ndim, feature_ndim, lambda t: conv_on_bias(t, pos=True)
        ) + _apply_feature_op(bounds.bias_upper, output_ndim, feature_ndim, lambda t: conv_on_bias(t, pos=False))
        new_bias_upper = _apply_feature_op(
            bounds.bias_upper, output_ndim, feature_ndim, lambda t: conv_on_bias(t, pos=True)
        ) + _apply_feature_op(bounds.bias_lower, output_ndim, feature_ndim, lambda t: conv_on_bias(t, pos=False))

        if bias is not None:
            # bias shape (C_out,); broadcast over (*batch, ·, H_out, W_out).
            new_bias_lower = new_bias_lower + bias.view(-1, 1, 1)
            new_bias_upper = new_bias_upper + bias.view(-1, 1, 1)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=new_linear_lower,
            bias_lower=new_bias_lower,
            linear_upper=new_linear_upper,
            bias_upper=new_bias_upper,
            input_ids=bounds.input_ids,
        )


def _resolve_conv2d_params(
    node: fx.Node,
    ctx: PropagationContext,
    args: tuple,
    kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise TypeError(f"Expected node.target to be str for call_module, got {type(node.target)}")
        module = ctx.get_module(node.target)
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"ForwardLBPConv2d expected nn.Conv2d module, got {type(module).__name__}")
        return (
            module.weight,
            module.bias,
            {
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
            },
        )
    weight = args[1] if len(args) > 1 else kwargs["weight"]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return (
        weight,
        bias,
        {
            "stride": args[3] if len(args) > 3 else kwargs.get("stride", 1),
            "padding": args[4] if len(args) > 4 else kwargs.get("padding", 0),
            "dilation": args[5] if len(args) > 5 else kwargs.get("dilation", 1),
            "groups": args[6] if len(args) > 6 else kwargs.get("groups", 1),
        },
    )


# ----------------------------------------------------------------------
# AvgPool2d
# ----------------------------------------------------------------------


class ForwardLBPAvgPool2d(ForwardLBPStrategy):
    """Forward-LBP strategy for ``nn.AvgPool2d`` / ``F.avg_pool2d`` and their
    adaptive variants.

    Average pooling is linear with non-negative weights (``1/k^2``), so the
    forward pass applies the same pool op to each bound (no sign decomposition).
    """

    def propagate_forward(self, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPAvgPool2d requires input to be LinearBounds")

        output_ndim = bounds.bias_lower.ndim
        feature_ndim = 3

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            if isinstance(module, nn.AdaptiveAvgPool2d):
                size = module.output_size
                apply_fn = lambda t: F.adaptive_avg_pool2d(t, size)  # noqa: E731
            elif isinstance(module, nn.AvgPool2d):
                pool_kwargs = _avg_pool_kwargs_from_module(module)
                apply_fn = lambda t: F.avg_pool2d(t, **pool_kwargs)  # noqa: E731
            else:
                raise TypeError(f"ForwardLBPAvgPool2d got unexpected module type {type(module).__name__}")
        elif node.target is F.adaptive_avg_pool2d:
            output_size = args[1] if len(args) > 1 else kwargs["output_size"]
            apply_fn = lambda t: F.adaptive_avg_pool2d(t, output_size)  # noqa: E731
        else:
            pool_kwargs = _avg_pool_kwargs_from_args(args, kwargs)
            apply_fn = lambda t: F.avg_pool2d(t, **pool_kwargs)  # noqa: E731

        new_linear_lower: list[LinearOperator] = []
        new_linear_upper: list[LinearOperator] = []
        for lower_op, upper_op in zip(bounds.linear_lowers_op, bounds.linear_uppers_op, strict=True):
            lower_t = lower_op.to_dense().tensor
            upper_t = upper_op.to_dense().tensor
            new_lower = _apply_feature_op(lower_t, output_ndim, feature_ndim, apply_fn)
            new_upper = _apply_feature_op(upper_t, output_ndim, feature_ndim, apply_fn)
            new_output_shape = (
                new_lower.shape[: output_ndim - feature_ndim]
                + new_lower.shape[output_ndim - feature_ndim : output_ndim]
            )
            new_linear_lower.append(DenseOperator(new_lower, output_shape=new_output_shape))
            new_linear_upper.append(DenseOperator(new_upper, output_shape=new_output_shape))

        new_bias_lower = _apply_feature_op(bounds.bias_lower, output_ndim, feature_ndim, apply_fn)
        new_bias_upper = _apply_feature_op(bounds.bias_upper, output_ndim, feature_ndim, apply_fn)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=new_linear_lower,
            bias_lower=new_bias_lower,
            linear_upper=new_linear_upper,
            bias_upper=new_bias_upper,
            input_ids=bounds.input_ids,
        )


def _avg_pool_kwargs_from_module(module: nn.AvgPool2d) -> dict[str, Any]:
    return {
        "kernel_size": module.kernel_size,
        "stride": module.stride if module.stride is not None else module.kernel_size,
        "padding": module.padding,
        "ceil_mode": module.ceil_mode,
        "count_include_pad": module.count_include_pad,
        "divisor_override": module.divisor_override,
    }


def _avg_pool_kwargs_from_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
    stride = args[2] if len(args) > 2 else kwargs.get("stride")
    if stride is None:
        stride = kernel_size
    return {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": args[3] if len(args) > 3 else kwargs.get("padding", 0),
        "ceil_mode": args[4] if len(args) > 4 else kwargs.get("ceil_mode", False),
        "count_include_pad": args[5] if len(args) > 5 else kwargs.get("count_include_pad", True),
        "divisor_override": args[6] if len(args) > 6 else kwargs.get("divisor_override"),
    }


# ----------------------------------------------------------------------
# MaxPool2d
# ----------------------------------------------------------------------


class ForwardLBPMaxPool2d(ForwardLBPStrategy):
    """Forward-LBP strategy for ``nn.MaxPool2d`` / ``F.max_pool2d``.

    Uses argmax-of-lower winner routing plus a slack term folded into the
    upper-bound bias, with optional alpha-CROWN interpolation between
    winner-routing (``alpha=1``) and pure IBP (``alpha=0``)::

        y_lower ≥ alpha_l · x[i*] + (1 − alpha_l) · max_lower
        y_upper ≤ alpha_u · x[i*] + max_upper − alpha_u · max_lower

    Implementation-wise: we first concretize the input to interval bounds,
    compute ``indices = argmax(lower_in, pool_window)``, then route each
    linear term through those indices (select one input position per output
    cell, scaled by alpha).
    """

    def propagate_forward(self, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMaxPool2d requires input to be LinearBounds")

        pool_kwargs = _resolve_maxpool2d_params(node, ctx, args, kwargs, input_bias_shape=bounds.bias_lower.shape)
        if pool_kwargs.pop("ceil_mode"):
            raise NotImplementedError("ForwardLBPMaxPool2d does not support ceil_mode=True")

        concrete = bounds.concretize()
        lower_in = concrete.lower
        upper_in = concrete.upper

        max_lower, indices = F.max_pool2d(
            lower_in,
            pool_kwargs["kernel_size"],
            pool_kwargs["stride"],
            pool_kwargs["padding"],
            pool_kwargs["dilation"],
            ceil_mode=False,
            return_indices=True,
        )
        max_upper = F.max_pool2d(
            upper_in,
            pool_kwargs["kernel_size"],
            pool_kwargs["stride"],
            pool_kwargs["padding"],
            pool_kwargs["dilation"],
            ceil_mode=False,
        )

        alpha_lower, alpha_upper = resolve_maxpool2d_alphas(
            ctx.alpha_provider,
            node,
            output_shape=max_lower.shape,
            device=lower_in.device,
            dtype=lower_in.dtype,
        )
        alpha_l = alpha_lower if alpha_lower is not None else torch.ones_like(max_lower)
        alpha_u = alpha_upper if alpha_upper is not None else torch.ones_like(max_upper)

        # Bias contributions:
        #   new_bias_lower = alpha_l · bias_in[i*] + (1 − alpha_l) · max_lower
        #   new_bias_upper = alpha_u · bias_in[i*] + (max_upper − alpha_u · max_lower)
        bias_lower_at_winner = _route_bias_via_indices(bounds.bias_lower, indices, max_lower.shape)
        bias_upper_at_winner = _route_bias_via_indices(bounds.bias_upper, indices, max_upper.shape)
        new_bias_lower = alpha_l * bias_lower_at_winner + (1.0 - alpha_l) * max_lower
        new_bias_upper = alpha_u * bias_upper_at_winner + (max_upper - alpha_u * max_lower)

        # Linear contributions: for each linear term, gather the coefficient at
        # the winner spatial position and scale by alpha.
        new_linear_lower: list[LinearOperator] = []
        new_linear_upper: list[LinearOperator] = []
        for lower_op, upper_op in zip(bounds.linear_lowers_op, bounds.linear_uppers_op, strict=True):
            lower_t = lower_op.to_dense().tensor
            upper_t = upper_op.to_dense().tensor
            routed_lower = _route_linear_via_indices(lower_t, indices, max_lower.shape)
            routed_upper = _route_linear_via_indices(upper_t, indices, max_upper.shape)
            # Slope is alpha (scalar) × 1 at i* (routed). Broadcast alpha over input axes.
            out_ndim = len(max_lower.shape)
            input_ndim = routed_lower.ndim - out_ndim
            alpha_l_bc = alpha_l.reshape(alpha_l.shape + (1,) * input_ndim)
            alpha_u_bc = alpha_u.reshape(alpha_u.shape + (1,) * input_ndim)
            new_linear_lower.append(DenseOperator(alpha_l_bc * routed_lower, output_shape=max_lower.shape))
            new_linear_upper.append(DenseOperator(alpha_u_bc * routed_upper, output_shape=max_upper.shape))

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=new_linear_lower,
            bias_lower=new_bias_lower,
            linear_upper=new_linear_upper,
            bias_upper=new_bias_upper,
            input_ids=bounds.input_ids,
        )


def _resolve_maxpool2d_params(
    node: fx.Node,
    ctx: PropagationContext,
    args: tuple,
    kwargs: dict,
    *,
    input_bias_shape: torch.Size,
) -> dict[str, Any]:
    if node.op == "call_module":
        module = ctx.get_module(node.target)
        if isinstance(module, nn.AdaptiveMaxPool2d):
            return _adaptive_to_fixed_maxpool(input_bias_shape, module.output_size)
        if not isinstance(module, nn.MaxPool2d):
            raise TypeError(f"ForwardLBPMaxPool2d got unexpected module type {type(module).__name__}")
        return {
            "kernel_size": _pair(module.kernel_size),
            "stride": _pair(module.stride if module.stride is not None else module.kernel_size),
            "padding": _pair(module.padding),
            "dilation": _pair(module.dilation),
            "ceil_mode": module.ceil_mode,
        }
    if node.target is F.adaptive_max_pool2d:
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        return _adaptive_to_fixed_maxpool(input_bias_shape, output_size)
    kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
    stride = args[2] if len(args) > 2 else kwargs.get("stride")
    if stride is None:
        stride = kernel_size
    return {
        "kernel_size": _pair(kernel_size),
        "stride": _pair(stride),
        "padding": _pair(args[3] if len(args) > 3 else kwargs.get("padding", 0)),
        "dilation": _pair(args[4] if len(args) > 4 else kwargs.get("dilation", 1)),
        "ceil_mode": bool(args[5] if len(args) > 5 else kwargs.get("ceil_mode", False)),
    }


def _adaptive_to_fixed_maxpool(
    input_shape: torch.Size, output_size: int | tuple[int, int] | tuple[int | None, int | None]
) -> dict[str, Any]:
    h_in, w_in = int(input_shape[-2]), int(input_shape[-1])
    if isinstance(output_size, int):
        h_out, w_out = output_size, output_size
    else:
        h_out = h_in if output_size[0] is None else int(output_size[0])
        w_out = w_in if output_size[1] is None else int(output_size[1])
    if h_in % h_out != 0 or w_in % w_out != 0:
        raise NotImplementedError(
            "ForwardLBPMaxPool2d for adaptive_max_pool2d requires input spatial dims divisible by output_size; "
            f"got input={(h_in, w_in)}, output={(h_out, w_out)}"
        )
    return {
        "kernel_size": (h_in // h_out, w_in // w_out),
        "stride": (h_in // h_out, w_in // w_out),
        "padding": (0, 0),
        "dilation": (1, 1),
        "ceil_mode": False,
    }


def _route_bias_via_indices(bias: torch.Tensor, indices: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    """Gather bias values at the winner indices.

    ``bias`` has shape ``(*batch, C, H_in, W_in)`` (same as the conv input);
    ``indices`` has shape ``(*batch, C, H_out, W_out)`` with flat spatial
    indices in ``[0, H_in * W_in)``; returns ``(*batch, C, H_out, W_out)``.
    """
    h_in, w_in = bias.shape[-2], bias.shape[-1]
    bias_flat = bias.reshape(*bias.shape[:-2], h_in * w_in)
    gathered = bias_flat.gather(dim=-1, index=indices.reshape(*indices.shape[:-2], -1))
    return gathered.reshape(output_shape)


def _route_linear_via_indices(linear: torch.Tensor, indices: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    """Gather linear-coefficient slices at the winner indices.

    ``linear`` has shape ``(*batch, C, H_in, W_in, *input_shape)``; ``indices``
    has shape ``(*batch, C, H_out, W_out)`` with flat indices into the spatial
    ``H_in * W_in`` axis. Returns ``(*batch, C, H_out, W_out, *input_shape)``.
    """
    indices_ndim = len(indices.shape)  # = ndim of (*batch, C, H_out, W_out)
    # Spatial axes of linear live at positions (indices_ndim - 2, indices_ndim - 1).
    spatial_axis = indices_ndim - 2
    batch_and_channel_shape = linear.shape[:spatial_axis]
    h_in = linear.shape[spatial_axis]
    w_in = linear.shape[spatial_axis + 1]
    input_shape = linear.shape[spatial_axis + 2 :]

    # Flatten (H_in, W_in) to a single axis: (*batch, C, H_in*W_in, *input_shape).
    linear_flat = linear.reshape(*batch_and_channel_shape, h_in * w_in, *input_shape)

    # Expand indices to broadcast over (*input_shape), with spatial flattened too.
    h_out, w_out = indices.shape[-2], indices.shape[-1]
    idx_flat = indices.reshape(*indices.shape[:-2], h_out * w_out)  # (*batch, C, H_out*W_out)
    idx_expanded = idx_flat.reshape(*idx_flat.shape, *([1] * len(input_shape))).expand(*idx_flat.shape, *input_shape)

    # Gather along the flattened spatial axis.
    gathered = linear_flat.gather(dim=spatial_axis, index=idx_expanded)

    # Unflatten (H_out*W_out) → (H_out, W_out).
    return gathered.reshape(*batch_and_channel_shape, h_out, w_out, *input_shape)


__all__ = [
    "ForwardLBPAvgPool2d",
    "ForwardLBPConv2d",
    "ForwardLBPMaxPool2d",
]
