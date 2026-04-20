from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ..context import PropagationContext


class IBPCat(ForwardIBPStrategy):
    """IBP strategy for cat."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        if not isinstance(tensors, (list, tuple)):
            raise TypeError("IBPCat expects first argument to be a list/tuple of tensors")

        for i, b in enumerate(tensors):
            if not isinstance(b, IntervalBounds):
                raise TypeError(f"IBPCat requires all inputs to be IntervalBounds, but input {i} is {type(b)}")

        lower = torch.cat([b.lower for b in tensors], dim=dim)
        upper = torch.cat([b.upper for b in tensors], dim=dim)

        batch_ndim = max(b.batch_ndim for b in tensors)
        return IntervalBounds(lower, upper, batch_ndim=min(batch_ndim, lower.ndim))


class IBPFlatten(ForwardIBPStrategy):
    """IBP strategy for flatten."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPFlatten requires input to be IntervalBounds")

        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise TypeError(f"Expected node.target to be str for call_module, got {type(node.target)}")

            module = ctx.get_module(node.target)
            start_dim: int = module.start_dim  # ty:ignore[invalid-assignment]
            end_dim: int = module.end_dim  # ty:ignore[invalid-assignment]
        else:
            start_dim: int = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
            end_dim: int = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)

        lower = torch.flatten(x_bounds.lower, start_dim, end_dim)
        upper = torch.flatten(x_bounds.upper, start_dim, end_dim)

        # Compute output batch_ndim: dims < start_dim (normalized) are untouched.
        # If start_dim falls inside batch dims, batch shrinks to start_dim; else preserved.
        normalized_start = start_dim if start_dim >= 0 else start_dim + x_bounds.lower.ndim
        new_batch_ndim = min(x_bounds.batch_ndim, normalized_start)
        return IntervalBounds(lower, upper, batch_ndim=new_batch_ndim)


class IBPGetItem(ForwardIBPStrategy):
    """IBP strategy for getitem (indexing/slicing)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPGetItem requires the input to be an IntervalBounds")

        item = args[1]

        return x_bounds[item]


class IBPPermute(ForwardIBPStrategy):
    """IBP strategy for permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPermute requires the input to be an IntervalBounds")

        # torch.permute(x, dims) → args[1] is a tuple
        # x.permute(*dims) → args[1:] are individual ints
        if len(args) > 2:
            dims = tuple(args[1:])
        else:
            dims = args[1]

        lower = x_bounds.lower.permute(dims)
        upper = x_bounds.upper.permute(dims)

        return IntervalBounds(lower, upper, batch_ndim=min(x_bounds.batch_ndim, lower.ndim))


class IBPReshape(ForwardIBPStrategy):
    """IBP strategy for reshape."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPReshape requires input to be IntervalBounds")

        # torch.reshape(x, shape) → args[1] is a tuple
        # x.reshape(*shape) → args[1:] are individual ints
        if len(args) > 2 or (len(args) == 2 and isinstance(args[1], int)):
            shape = tuple(args[1:])
        else:
            shape = args[1]

        lower = x_bounds.lower.reshape(shape)
        upper = x_bounds.upper.reshape(shape)

        return IntervalBounds(lower, upper, batch_ndim=min(x_bounds.batch_ndim, lower.ndim))


class IBPSelect(ForwardIBPStrategy):
    """IBP strategy for select."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSelect requires the input to be an IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index")

        if index is None:
            raise ValueError("select requires an index argument")

        sel_lower = x_bounds.lower.select(dim=dim, index=index)
        normalized_dim = dim if dim >= 0 else dim + x_bounds.lower.ndim
        new_batch_ndim = x_bounds.batch_ndim - 1 if normalized_dim < x_bounds.batch_ndim else x_bounds.batch_ndim
        return IntervalBounds(
            sel_lower,
            x_bounds.upper.select(dim=dim, index=index),
            batch_ndim=min(new_batch_ndim, sel_lower.ndim),
        )


class IBPSqueeze(ForwardIBPStrategy):
    """IBP strategy for squeeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqueeze requires input to be IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", None)

        lower = x_bounds.lower.squeeze(dim=dim)
        upper = x_bounds.upper.squeeze(dim=dim)

        return IntervalBounds(lower, upper, batch_ndim=min(x_bounds.batch_ndim, lower.ndim))


class IBPStack(ForwardIBPStrategy):
    """IBP strategy for stack."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        if not isinstance(tensors, (list, tuple)):
            raise TypeError("IBPStack expects first argument to be a list/tuple of tensors")

        for i, b in enumerate(tensors):
            if not isinstance(b, IntervalBounds):
                raise TypeError(f"IBPStack requires all inputs to be IntervalBounds, but input {i} is {type(b)}")

        lower = torch.stack([b.lower for b in tensors], dim=dim)
        upper = torch.stack([b.upper for b in tensors], dim=dim)

        input_batch_ndim = max(b.batch_ndim for b in tensors)
        normalized_dim = dim if dim >= 0 else dim + lower.ndim
        new_batch_ndim = input_batch_ndim + 1 if normalized_dim <= input_batch_ndim else input_batch_ndim
        return IntervalBounds(lower, upper, batch_ndim=min(new_batch_ndim, lower.ndim))


class IBPTranspose(ForwardIBPStrategy):
    """IBP strategy for transpose."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTranspose requires the input to be an IntervalBounds")

        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0")
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1")

        if dim0 is None or dim1 is None:
            raise ValueError("transpose requires dim0 and dim1 arguments")

        lower = x_bounds.lower.transpose(dim0, dim1)
        upper = x_bounds.upper.transpose(dim0, dim1)

        return IntervalBounds(lower, upper, batch_ndim=min(x_bounds.batch_ndim, lower.ndim))


class IBPUnsqueeze(ForwardIBPStrategy):
    """IBP strategy for unsqueeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPUnsqueeze requires input to be IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        if dim is None:
            raise ValueError("unsqueeze requires a dim argument")

        lower = x_bounds.lower.unsqueeze(dim=dim)
        upper = x_bounds.upper.unsqueeze(dim=dim)

        normalized_dim = dim if dim >= 0 else dim + lower.ndim
        new_batch_ndim = x_bounds.batch_ndim + 1 if normalized_dim <= x_bounds.batch_ndim else x_bounds.batch_ndim
        return IntervalBounds(lower, upper, batch_ndim=min(new_batch_ndim, lower.ndim))


class IBPView(ForwardIBPStrategy):
    """IBP strategy for view."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPView requires input to be IntervalBounds")

        # x.view(*shape) → args[1:] are individual ints
        # x.view(shape) → args[1] is a tuple
        if len(args) > 2 or (len(args) == 2 and isinstance(args[1], int)):
            size = tuple(args[1:])
        else:
            size = args[1]

        lower = x_bounds.lower.view(size)
        upper = x_bounds.upper.view(size)

        return IntervalBounds(lower, upper, batch_ndim=min(x_bounds.batch_ndim, lower.ndim))
