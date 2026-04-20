from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ...linear_operators import LinearOperator, ReshapeOperator
from .base import ForwardLBPStrategy
from .utils import transform_linear_terms

if TYPE_CHECKING:
    import torch.fx as fx

    from ...regions import SimpleRegion
    from ..context import PropagationContext


def _reshape_ops(ops: list[LinearOperator], new_output_shape: tuple[int, ...]) -> list[LinearOperator]:
    """Wrap each operator in a :class:`ReshapeOperator` presenting ``new_output_shape``.

    Operators whose ``output_shape`` already matches ``new_output_shape`` are
    returned unchanged. This keeps structured operators (e.g.
    :class:`Conv2dPatchOperator`) unmaterialized across reshape-family ops
    (``flatten``/``view``/``reshape``/``squeeze``/``unsqueeze``).
    """
    target = torch.Size(new_output_shape)
    result: list[LinearOperator] = []
    for op in ops:
        if op.output_shape == target:
            result.append(op)
        else:
            result.append(ReshapeOperator(op, target))
    return result


class ForwardLBPConcat(ForwardLBPStrategy):
    """Forward LBP strategy for torch.cat."""

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
                raise TypeError(f"ForwardLBPConcat requires all inputs to be LinearBounds, but input {i} is {type(t)}")
            bounds_list.append(t)

        # Collect ordered unique input_ids and their associated regions.
        all_regions: dict[int, SimpleRegion] = {}
        for b in bounds_list:
            for input_id, region in zip(b.input_ids, b.regions, strict=True):
                if input_id not in all_regions:
                    all_regions[input_id] = region
                elif all_regions[input_id].shape != region.shape:  # type: ignore[union-attr]
                    raise ValueError(
                        f"Cannot cat: input_id {input_id} appears with different region shapes "
                        f"{all_regions[input_id].shape} vs {region.shape}"  # type: ignore[union-attr]
                    )

        ordered_ids = list(all_regions.keys())
        regions: list[SimpleRegion] = list(all_regions.values())

        # Build a lookup {input_id: index} for each bounds object (O(1) per lookup).
        id_index: list[dict[int, int]] = [{input_id: i for i, input_id in enumerate(b.input_ids)} for b in bounds_list]

        linear_lower: list[torch.Tensor] = []
        linear_upper: list[torch.Tensor] = []
        for input_id, region in zip(ordered_ids, regions, strict=True):
            expected_input_axes: torch.Size | None = None
            lower_parts: list[torch.Tensor] = []
            upper_parts: list[torch.Tensor] = []
            for b, lookup in zip(bounds_list, id_index, strict=True):
                if input_id in lookup:
                    idx = lookup[input_id]
                    lower_linear = b.linear_lowers[idx]
                    upper_linear = b.linear_uppers[idx]
                    lower_input_axes = lower_linear.shape[len(b.bias_lower.shape) :]
                    upper_input_axes = upper_linear.shape[len(b.bias_upper.shape) :]
                    if lower_input_axes != upper_input_axes:
                        raise ValueError(
                            f"Lower and upper input axes must match for input_id {input_id}: "
                            f"{tuple(lower_input_axes)} vs {tuple(upper_input_axes)}"
                        )
                    if expected_input_axes is None:
                        expected_input_axes = lower_input_axes
                    elif expected_input_axes != lower_input_axes:
                        raise ValueError(
                            f"Inconsistent input axes for input_id {input_id}: "
                            f"expected {tuple(expected_input_axes)}, got {tuple(lower_input_axes)}"
                        )
                    lower_parts.append(lower_linear)
                    upper_parts.append(upper_linear)
                else:
                    if expected_input_axes is None:
                        expected_input_axes = torch.Size(region.shape)
                    zeros = torch.zeros(
                        *b.bias_lower.shape,
                        *expected_input_axes,
                        dtype=b.bias_lower.dtype,
                        device=b.bias_lower.device,
                    )
                    lower_parts.append(zeros)
                    upper_parts.append(zeros)
            linear_lower.append(torch.cat(lower_parts, dim=dim))
            linear_upper.append(torch.cat(upper_parts, dim=dim))

        return LinearBounds(
            regions=regions,
            linear_lower=linear_lower,
            bias_lower=torch.cat([b.bias_lower for b in bounds_list], dim=dim),
            linear_upper=linear_upper,
            bias_upper=torch.cat([b.bias_upper for b in bounds_list], dim=dim),
            input_ids=ordered_ids if ordered_ids else None,
        )


class ForwardLBPFlatten(ForwardLBPStrategy):
    """Forward LBP strategy for flatten."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPFlatten requires input to be LinearBounds")

        start_dim = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        end_dim = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)

        output_ndim = bounds.bias_lower.ndim
        if start_dim < 0:
            start_dim += output_ndim
        if end_dim < 0:
            end_dim += output_ndim

        if start_dim < 0 or start_dim >= output_ndim or end_dim < 0 or end_dim >= output_ndim:
            raise ValueError(
                f"flatten dims must be in [0, {output_ndim - 1}], got start_dim={start_dim}, end_dim={end_dim}"
            )

        if end_dim < start_dim:
            raise ValueError(f"flatten end_dim must be >= start_dim, got start_dim={start_dim}, end_dim={end_dim}")

        bias_lower = bounds.bias_lower.flatten(start_dim, end_dim)
        bias_upper = bounds.bias_upper.flatten(start_dim, end_dim)
        new_output_shape = tuple(bias_lower.shape)
        linear_lower = _reshape_ops(bounds.linear_lowers_op, new_output_shape)
        linear_upper = _reshape_ops(bounds.linear_uppers_op, new_output_shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPGetItem(ForwardLBPStrategy):
    """Forward LBP strategy for getitem (operator.getitem)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]
        index = args[1]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPGetItem requires input to be LinearBounds")

        return bounds[index]


class ForwardLBPReshape(ForwardLBPStrategy):
    """Forward LBP strategy for reshape."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPReshape requires input to be LinearBounds")

        if len(args) == 2 and isinstance(args[1], (tuple, list, torch.Size)):
            target_shape = tuple(args[1])
        else:
            target_shape = tuple(args[1:])

        bias_lower = bounds.bias_lower.reshape(target_shape)
        bias_upper = bounds.bias_upper.reshape(target_shape)
        new_output_shape = tuple(bias_lower.shape)
        linear_lower = _reshape_ops(bounds.linear_lowers_op, new_output_shape)
        linear_upper = _reshape_ops(bounds.linear_uppers_op, new_output_shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPSelect(ForwardLBPStrategy):
    """Forward LBP strategy for select."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSelect requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index", 0)

        output_ndim = bounds.bias_lower.ndim
        if dim < 0:
            dim += output_ndim
        if dim < 0 or dim >= output_ndim:
            raise ValueError(f"select dim must be in [0, {output_ndim - 1}], got {dim}")

        linear_lower = transform_linear_terms(bounds.linear_lowers, lambda linear: linear.select(dim, index))
        linear_upper = transform_linear_terms(bounds.linear_uppers, lambda linear: linear.select(dim, index))
        bias_lower = bounds.bias_lower.select(dim, index)
        bias_upper = bounds.bias_upper.select(dim, index)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPSqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for squeeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSqueeze requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        output_ndim = bounds.bias_lower.ndim

        if dim is not None:
            if dim < 0:
                dim += output_ndim
            if dim < 0 or dim >= output_ndim:
                raise ValueError(f"squeeze dim must be in [0, {output_ndim - 1}], got {dim}")

            bias_lower = bounds.bias_lower.squeeze(dim)
            bias_upper = bounds.bias_upper.squeeze(dim)
        else:
            target_shape = tuple(size for size in bounds.bias_lower.shape if size != 1)
            bias_lower = bounds.bias_lower.reshape(target_shape)
            bias_upper = bounds.bias_upper.reshape(target_shape)

        new_output_shape = tuple(bias_lower.shape)
        linear_lower = _reshape_ops(bounds.linear_lowers_op, new_output_shape)
        linear_upper = _reshape_ops(bounds.linear_uppers_op, new_output_shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


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

        # Collect ordered unique input_ids and their associated regions.
        all_regions: dict[int, SimpleRegion] = {}
        for b in bounds_list:
            for input_id, region in zip(b.input_ids, b.regions, strict=True):
                if input_id not in all_regions:
                    all_regions[input_id] = region
                elif all_regions[input_id].shape != region.shape:  # type: ignore[union-attr]
                    raise ValueError(
                        f"Cannot stack: input_id {input_id} appears with different region shapes "
                        f"{all_regions[input_id].shape} vs {region.shape}"  # type: ignore[union-attr]
                    )

        ordered_ids = list(all_regions.keys())
        regions: list[SimpleRegion] = list(all_regions.values())

        # Build a lookup {input_id: index} for each bounds object.
        id_index: list[dict[int, int]] = [{input_id: i for i, input_id in enumerate(b.input_ids)} for b in bounds_list]

        linear_lower: list[torch.Tensor] = []
        linear_upper: list[torch.Tensor] = []
        for input_id, region in zip(ordered_ids, regions, strict=True):
            expected_input_axes: torch.Size | None = None
            lower_parts: list[torch.Tensor] = []
            upper_parts: list[torch.Tensor] = []
            for b, lookup in zip(bounds_list, id_index, strict=True):
                if input_id in lookup:
                    idx = lookup[input_id]
                    lower_linear = b.linear_lowers[idx]
                    upper_linear = b.linear_uppers[idx]
                    lower_input_axes = lower_linear.shape[len(b.bias_lower.shape) :]
                    upper_input_axes = upper_linear.shape[len(b.bias_upper.shape) :]
                    if lower_input_axes != upper_input_axes:
                        raise ValueError(
                            f"Lower and upper input axes must match for input_id {input_id}: "
                            f"{tuple(lower_input_axes)} vs {tuple(upper_input_axes)}"
                        )
                    if expected_input_axes is None:
                        expected_input_axes = lower_input_axes
                    elif expected_input_axes != lower_input_axes:
                        raise ValueError(
                            f"Inconsistent input axes for input_id {input_id}: "
                            f"expected {tuple(expected_input_axes)}, got {tuple(lower_input_axes)}"
                        )
                    lower_parts.append(lower_linear)
                    upper_parts.append(upper_linear)
                else:
                    if expected_input_axes is None:
                        expected_input_axes = torch.Size(region.shape)
                    zeros = torch.zeros(
                        *b.bias_lower.shape,
                        *expected_input_axes,
                        dtype=b.bias_lower.dtype,
                        device=b.bias_lower.device,
                    )
                    lower_parts.append(zeros)
                    upper_parts.append(zeros)
            linear_lower.append(torch.stack(lower_parts, dim=dim))
            linear_upper.append(torch.stack(upper_parts, dim=dim))

        return LinearBounds(
            regions=regions,
            linear_lower=linear_lower,
            bias_lower=torch.stack([b.bias_lower for b in bounds_list], dim=dim),
            linear_upper=linear_upper,
            bias_upper=torch.stack([b.bias_upper for b in bounds_list], dim=dim),
            input_ids=ordered_ids if ordered_ids else None,
        )


class ForwardLBPTranspose(ForwardLBPStrategy):
    """Forward LBP strategy for transpose and permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPTranspose requires input to be LinearBounds")

        # torch.transpose(input, dim0, dim1)
        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0", 0)
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 1)

        output_ndim = bounds.bias_lower.ndim
        if dim0 < 0:
            dim0 += output_ndim
        if dim1 < 0:
            dim1 += output_ndim

        if dim0 < 0 or dim0 >= output_ndim or dim1 < 0 or dim1 >= output_ndim:
            raise ValueError(f"transpose dims must be in [0, {output_ndim - 1}], got dim0={dim0}, dim1={dim1}")

        linear_lower = transform_linear_terms(bounds.linear_lowers, lambda linear: linear.transpose(dim0, dim1))
        linear_upper = transform_linear_terms(bounds.linear_uppers, lambda linear: linear.transpose(dim0, dim1))
        bias_lower = bounds.bias_lower.transpose(dim0, dim1)
        bias_upper = bounds.bias_upper.transpose(dim0, dim1)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPPermute(ForwardLBPStrategy):
    """Forward LBP strategy for permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPPermute requires input to be LinearBounds")

        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            dims = tuple(args[1])
        else:
            dims = tuple(args[1:])

        output_ndim = bounds.bias_lower.ndim
        if len(dims) != output_ndim:
            raise ValueError(f"permute expects {output_ndim} dims, got {len(dims)}")

        dims = tuple(d + output_ndim if d < 0 else d for d in dims)
        if sorted(dims) != list(range(output_ndim)):
            raise ValueError(f"invalid permutation for {output_ndim} dims: {dims}")

        linear_lower = transform_linear_terms(
            bounds.linear_lowers,
            lambda linear: linear.permute(*dims, *range(output_ndim, linear.ndim)),
        )
        linear_upper = transform_linear_terms(
            bounds.linear_uppers,
            lambda linear: linear.permute(*dims, *range(output_ndim, linear.ndim)),
        )
        bias_lower = bounds.bias_lower.permute(*dims)
        bias_upper = bounds.bias_upper.permute(*dims)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPUnsqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for unsqueeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPUnsqueeze requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        output_ndim = bounds.bias_lower.ndim
        if dim < 0:
            dim += output_ndim + 1
        if dim < 0 or dim > output_ndim:
            raise ValueError(f"unsqueeze dim must be in [0, {output_ndim}], got {dim}")

        bias_lower = bounds.bias_lower.unsqueeze(dim)
        bias_upper = bounds.bias_upper.unsqueeze(dim)
        new_output_shape = tuple(bias_lower.shape)
        linear_lower = _reshape_ops(bounds.linear_lowers_op, new_output_shape)
        linear_upper = _reshape_ops(bounds.linear_uppers_op, new_output_shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )


class ForwardLBPView(ForwardLBPStrategy):
    """Forward LBP strategy for view."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPView requires input to be LinearBounds")

        if len(args) == 2 and isinstance(args[1], (tuple, list, torch.Size)):
            shape = tuple(args[1])
        else:
            shape = tuple(args[1:])

        bias_lower = bounds.bias_lower.view(*shape)
        bias_upper = bounds.bias_upper.view(*shape)
        new_output_shape = tuple(bias_lower.shape)
        linear_lower = _reshape_ops(bounds.linear_lowers_op, new_output_shape)
        linear_upper = _reshape_ops(bounds.linear_uppers_op, new_output_shape)

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
