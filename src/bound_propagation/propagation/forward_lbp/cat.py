from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ...regions import SimpleRegion
    from ..context import PropagationContext


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
            input_dim = torch.Size(region.shape).numel()  # type: ignore[union-attr]
            lower_parts: list[torch.Tensor] = []
            upper_parts: list[torch.Tensor] = []
            for b, lookup in zip(bounds_list, id_index, strict=True):
                if input_id in lookup:
                    idx = lookup[input_id]
                    lower_parts.append(b.linear_lowers[idx])
                    upper_parts.append(b.linear_uppers[idx])
                else:
                    zeros = torch.zeros(
                        *b.bias_lower.shape,
                        input_dim,
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
