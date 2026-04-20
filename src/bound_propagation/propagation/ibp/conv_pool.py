"""IBP strategies for 2D convolution and pooling operations.

Convolution, average pooling, and max pooling are all monotone (or signed-monotone)
operations on the input tensor, so interval bounds propagate through them using
the same sign-decomposition trick as :class:`IBPLinear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


# ----------------------------------------------------------------------
# Convolution
# ----------------------------------------------------------------------


class IBPConv2d(ForwardIBPStrategy):
    """IBP strategy for ``nn.Conv2d`` / ``F.conv2d``.

    The convolution ``y = conv2d(x, W) + b`` is linear; each output element is a
    weighted sum of a localized input patch. Per output element, positive weight
    entries pull the bound toward the input's matching bound and negative
    entries pull it toward the opposite bound. We therefore sign-decompose the
    kernel and apply two separate convolutions.
    """

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPConv2d requires input to be IntervalBounds")

        weight, bias, conv_kwargs = _resolve_conv2d_params(node, ctx, args, kwargs)

        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

        lower = F.conv2d(x_bounds.lower, weight_pos, bias=None, **conv_kwargs) + F.conv2d(
            x_bounds.upper, weight_neg, bias=None, **conv_kwargs
        )
        upper = F.conv2d(x_bounds.upper, weight_pos, bias=None, **conv_kwargs) + F.conv2d(
            x_bounds.lower, weight_neg, bias=None, **conv_kwargs
        )

        if bias is not None:
            # Broadcast against the trailing (C, H, W) axes regardless of whether
            # the input includes a leading batch dim.
            lower = lower + bias.view(-1, 1, 1)
            upper = upper + bias.view(-1, 1, 1)

        return IntervalBounds(lower, upper)


def _resolve_conv2d_params(
    node: fx.Node,
    ctx: PropagationContext,
    args: tuple,
    kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Extract weight, bias, and hyperparameter kwargs for conv2d.

    Handles both ``call_module`` (``nn.Conv2d``) and ``call_function``
    (``F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)``)
    invocations.
    """
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise TypeError(f"Expected node.target to be str for call_module, got {type(node.target)}")
        module = ctx.get_module(node.target)
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"IBPConv2d expected nn.Conv2d module, got {type(module).__name__}")
        weight = module.weight
        bias = module.bias
        conv_kwargs: dict[str, Any] = {
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
        }
    else:
        # F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        weight = args[1] if len(args) > 1 else kwargs["weight"]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        conv_kwargs = {
            "stride": args[3] if len(args) > 3 else kwargs.get("stride", 1),
            "padding": args[4] if len(args) > 4 else kwargs.get("padding", 0),
            "dilation": args[5] if len(args) > 5 else kwargs.get("dilation", 1),
            "groups": args[6] if len(args) > 6 else kwargs.get("groups", 1),
        }

    if weight.ndim != 4:
        raise ValueError(f"conv2d weight must be 4D (out_c, in_c/groups, kH, kW), got shape {tuple(weight.shape)}")

    return weight, bias, conv_kwargs


# ----------------------------------------------------------------------
# Average pooling
# ----------------------------------------------------------------------


class IBPAvgPool2d(ForwardIBPStrategy):
    """IBP strategy for ``nn.AvgPool2d`` / ``F.avg_pool2d`` / ``nn.AdaptiveAvgPool2d``.

    Average pooling is linear with all-positive weights (``1/k^2``), so bounds
    propagate monotonically: ``lower = avg_pool(x.lower)``, ``upper = avg_pool(x.upper)``.
    """

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPAvgPool2d requires input to be IntervalBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            if isinstance(module, nn.AdaptiveAvgPool2d):
                lower = F.adaptive_avg_pool2d(x_bounds.lower, module.output_size)
                upper = F.adaptive_avg_pool2d(x_bounds.upper, module.output_size)
                return IntervalBounds(lower, upper)
            if not isinstance(module, nn.AvgPool2d):
                raise TypeError(f"IBPAvgPool2d expected nn.AvgPool2d module, got {type(module).__name__}")
            pool_kwargs = _avg_pool_kwargs_from_module(module)
        elif node.target is F.adaptive_avg_pool2d:
            output_size = args[1] if len(args) > 1 else kwargs["output_size"]
            lower = F.adaptive_avg_pool2d(x_bounds.lower, output_size)
            upper = F.adaptive_avg_pool2d(x_bounds.upper, output_size)
            return IntervalBounds(lower, upper)
        else:
            # F.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
            #              count_include_pad=True, divisor_override=None)
            pool_kwargs = _avg_pool_kwargs_from_args(args, kwargs)

        lower = F.avg_pool2d(x_bounds.lower, **pool_kwargs)
        upper = F.avg_pool2d(x_bounds.upper, **pool_kwargs)
        return IntervalBounds(lower, upper)


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
# Max pooling
# ----------------------------------------------------------------------


class IBPMaxPool2d(ForwardIBPStrategy):
    """IBP strategy for ``nn.MaxPool2d`` / ``F.max_pool2d`` / ``nn.AdaptiveMaxPool2d``.

    Max pooling is monotone: ``max(a_i) <= max(b_i)`` when ``a_i <= b_i`` for all
    ``i``, so bounds propagate by pooling the interval endpoints separately.
    """

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPMaxPool2d requires input to be IntervalBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            if isinstance(module, nn.AdaptiveMaxPool2d):
                lower = F.adaptive_max_pool2d(x_bounds.lower, module.output_size)
                upper = F.adaptive_max_pool2d(x_bounds.upper, module.output_size)
                return IntervalBounds(lower, upper)
            if not isinstance(module, nn.MaxPool2d):
                raise TypeError(f"IBPMaxPool2d expected nn.MaxPool2d module, got {type(module).__name__}")
            pool_kwargs = _max_pool_kwargs_from_module(module)
        elif node.target is F.adaptive_max_pool2d:
            output_size = args[1] if len(args) > 1 else kwargs["output_size"]
            lower = F.adaptive_max_pool2d(x_bounds.lower, output_size)
            upper = F.adaptive_max_pool2d(x_bounds.upper, output_size)
            return IntervalBounds(lower, upper)
        else:
            pool_kwargs = _max_pool_kwargs_from_args(args, kwargs)

        lower = F.max_pool2d(x_bounds.lower, **pool_kwargs)
        upper = F.max_pool2d(x_bounds.upper, **pool_kwargs)
        return IntervalBounds(lower, upper)


def _max_pool_kwargs_from_module(module: nn.MaxPool2d) -> dict[str, Any]:
    return {
        "kernel_size": module.kernel_size,
        "stride": module.stride if module.stride is not None else module.kernel_size,
        "padding": module.padding,
        "dilation": module.dilation,
        "ceil_mode": module.ceil_mode,
    }


def _max_pool_kwargs_from_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    # F.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
    #              ceil_mode=False, return_indices=False)
    kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
    stride = args[2] if len(args) > 2 else kwargs.get("stride")
    if stride is None:
        stride = kernel_size
    return_indices = args[6] if len(args) > 6 else kwargs.get("return_indices", False)
    if return_indices:
        raise NotImplementedError("IBPMaxPool2d does not support return_indices=True")
    return {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": args[3] if len(args) > 3 else kwargs.get("padding", 0),
        "dilation": args[4] if len(args) > 4 else kwargs.get("dilation", 1),
        "ceil_mode": args[5] if len(args) > 5 else kwargs.get("ceil_mode", False),
    }


__all__ = [
    "IBPAvgPool2d",
    "IBPConv2d",
    "IBPMaxPool2d",
]
