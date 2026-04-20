"""Verify ``batch_ndim`` flows through the IBP propagator.

These tests lock in the Stage 2 contract: the propagator seeds placeholder
bounds with the caller-supplied ``batch_ndim`` and each IBP strategy
propagates it (possibly updating for shape ops) so the output
``IntervalBounds`` carries a correct ``batch_ndim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle


def _region_with_batch(shape_with_batch: tuple[int, ...], feature_ndim: int) -> HyperRectangle:
    lower = torch.zeros(*shape_with_batch)
    upper = torch.ones(*shape_with_batch)
    del feature_ndim
    return HyperRectangle(lower=lower, upper=upper)


def test_ibp_preserves_batch_ndim_through_linear_stack() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    dummy = torch.zeros(4)  # feature shape is (4,)
    bound_model = BoundModel(model, dummy_inputs=(dummy,), method="ibp")

    # Region with two leading batch dims: shape (3, 5, 4).
    region = _region_with_batch((3, 5, 4), feature_ndim=1)
    bounds = bound_model.propagate(region)

    assert bounds.shape == (3, 5, 2)
    assert bounds.batch_ndim == 2
    assert bounds.feature_shape == (2,)


def test_ibp_flatten_collapses_feature_dims_but_preserves_batch_ndim() -> None:
    class FlattenNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.flatten(x)

    model = FlattenNet()
    dummy = torch.zeros(3, 4)
    bound_model = BoundModel(model, dummy_inputs=(dummy,), method="ibp")

    region = _region_with_batch((2, 3, 4), feature_ndim=2)
    bounds = bound_model.propagate(region)

    assert bounds.shape == (2, 12)
    assert bounds.batch_ndim == 1
    assert bounds.feature_shape == (12,)


def test_ibp_batch_ndim_zero_when_no_batch_dims() -> None:
    model = nn.Linear(4, 2)
    dummy = torch.zeros(4)
    bound_model = BoundModel(model, dummy_inputs=(dummy,), method="ibp")

    region = _region_with_batch((4,), feature_ndim=1)
    bounds = bound_model.propagate(region)

    assert bounds.shape == (2,)
    assert bounds.batch_ndim == 0
    assert bounds.feature_shape == (2,)
