"""Shared infrastructure for shape-compatibility tests.

These tests verify that every :class:`BoundPropagator` subclass

* accepts the same set of input shapes that PyTorch documents for each op,
* produces bounds whose output shape matches eager-mode PyTorch, and
* produces bounds that numerically envelope the eager-mode output on a
  grid of samples + region corners + the region center.

Each test file declares the shapes PyTorch allows for that op, then the
harness exercises every method against each shape. Cross-method
divergences should be surfaced as test failures (or ``xfail`` with a
one-line reason) so they cannot drift silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from bound_propagation import BoundModel, HyperRectangle
from bound_propagation.bounds import IntervalBounds, LinearBounds

ALL_METHODS: tuple[str, ...] = (
    "ibp",
    "forward_lbp",
    "backward_lbp",
    "forward_backward_lbp",
    "crown_ibp",
)


def _enumerate_corners(region: HyperRectangle, max_corners: int = 32) -> torch.Tensor:
    """Sample up to *max_corners* corners of *region*.

    For low-dim regions we enumerate every corner; for high-dim regions
    (where 2**numel would overflow long before any test cares) we draw
    random {lower, upper} per-element masks.
    """
    flat_lower = region.lower.reshape(-1)
    flat_upper = region.upper.reshape(-1)
    dim = flat_lower.numel()
    if dim == 0:
        return region.lower.unsqueeze(0)
    if dim <= 20 and (1 << dim) <= max_corners:
        total = 1 << dim
        bits = torch.arange(total)
        masks = ((bits.unsqueeze(1) >> torch.arange(dim)) & 1).to(torch.bool)
    else:
        masks = torch.randint(0, 2, (max_corners, dim), dtype=torch.bool)
    corners = torch.where(masks, flat_upper, flat_lower)
    return corners.reshape(-1, *region.lower.shape)


def _sample_inputs(
    regions: Sequence[HyperRectangle],
    *,
    n_random: int = 32,
    seed: int = 0,
) -> list[list[torch.Tensor]]:
    torch.manual_seed(seed)
    per_region_batches: list[torch.Tensor] = []
    for region in regions:
        corners = _enumerate_corners(region)
        center = ((region.lower + region.upper) / 2).unsqueeze(0)
        rand = region.lower.unsqueeze(0) + torch.rand(
            (n_random, *region.lower.shape),
            dtype=region.lower.dtype,
        ) * (region.upper - region.lower).unsqueeze(0)
        per_region_batches.append(torch.cat([corners, center, rand], dim=0))

    n_total = min(b.shape[0] for b in per_region_batches)
    return [[b[i] for b in per_region_batches] for i in range(n_total)]


def _concretize(bounds: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(bounds, LinearBounds):
        return bounds.concretize()
    if isinstance(bounds, IntervalBounds):
        return bounds.lower, bounds.upper
    if hasattr(bounds, "concretize"):
        return bounds.concretize()
    raise TypeError(f"Cannot concretize bounds of type {type(bounds).__name__}")


def check_op_compatibility(
    fn: Callable[..., torch.Tensor],
    dummy_inputs: Sequence[torch.Tensor],
    regions: Sequence[HyperRectangle],
    method: str,
    *,
    atol: float = 1e-4,
    n_random: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run *fn* under *method* and verify shape + soundness.

    Returns the concretized ``(lower, upper)`` so callers can perform
    extra method-specific assertions if needed.
    """
    bm = BoundModel(fn, dummy_inputs=tuple(dummy_inputs), method=method)
    bounds = bm.propagate(*regions)
    lower, upper = _concretize(bounds)

    midpoint = [((r.lower + r.upper) / 2) for r in regions]
    expected = fn(*midpoint)

    assert lower.shape == expected.shape, (
        f"[method={method}] lower-bound shape {tuple(lower.shape)} does not match "
        f"eager output shape {tuple(expected.shape)}"
    )
    assert upper.shape == expected.shape, (
        f"[method={method}] upper-bound shape {tuple(upper.shape)} does not match "
        f"eager output shape {tuple(expected.shape)}"
    )
    assert torch.all(lower <= upper + atol), f"[method={method}] lower > upper somewhere: lower={lower}, upper={upper}"

    samples = _sample_inputs(regions, n_random=n_random, seed=seed)
    for row in samples:
        out = fn(*row)
        assert out.shape == expected.shape, (
            f"[method={method}] eager output shape varies across samples "
            f"({tuple(out.shape)} vs {tuple(expected.shape)})"
        )
        assert torch.all(lower <= out + atol), (
            f"[method={method}] lower bound violated:\n  inputs={row}\n  out={out}\n  lower={lower}"
        )
        assert torch.all(out <= upper + atol), (
            f"[method={method}] upper bound violated:\n  inputs={row}\n  out={out}\n  upper={upper}"
        )
    return lower, upper


def make_region(
    shape: tuple[int, ...],
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> HyperRectangle:
    """Build a uniform-width hyperrectangle of the requested feature shape."""
    if shape == ():
        return HyperRectangle(lower=torch.tensor(lower), upper=torch.tensor(upper))
    return HyperRectangle(
        lower=torch.full(shape, lower),
        upper=torch.full(shape, upper),
    )


def safe_positive_region(
    shape: tuple[int, ...],
    *,
    lower: float = 0.1,
    upper: float = 2.0,
) -> HyperRectangle:
    """Region with strictly-positive values (for log / sqrt / reciprocal)."""
    return make_region(shape, lower=lower, upper=upper)


__all__ = [
    "ALL_METHODS",
    "check_op_compatibility",
    "make_region",
    "safe_positive_region",
]
