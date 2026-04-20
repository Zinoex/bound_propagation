"""Facade tests for small CNN architectures across IBP / backward-LBP methods."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle


def _sample_sound(fn, region: HyperRectangle, lower: torch.Tensor, upper: torch.Tensor, n: int = 50) -> None:
    for _ in range(n):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = fn(x)
        assert torch.all(lower <= y + 1e-5), (lower, y)
        assert torch.all(y <= upper + 1e-5), (upper, y)


PROP_METHODS: list[Literal["ibp", "backward_lbp", "forward_lbp", "crown_ibp", "forward_backward_lbp"]] = [
    "ibp",
    "backward_lbp",
    "forward_lbp",
    "crown_ibp",
    "forward_backward_lbp",
]


@pytest.fixture
def small_cnn() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
    )


@pytest.fixture
def cnn_region() -> HyperRectangle:
    torch.manual_seed(1)
    lower = torch.randn(2, 8, 8)  # unbatched; facade infers batch_ndim=0
    upper = lower + 0.1
    return HyperRectangle(lower=lower, upper=upper)


@pytest.mark.parametrize("method", PROP_METHODS)
def test_cnn_end_to_end(
    small_cnn: nn.Module,
    cnn_region: HyperRectangle,
    method: Literal["ibp", "backward_lbp", "forward_lbp", "crown_ibp", "forward_backward_lbp"],
) -> None:
    dummy = torch.zeros(2, 8, 8)
    bm = BoundModel(small_cnn, dummy_inputs=(dummy,), method=method)
    bounds = bm.propagate(cnn_region)
    lower, upper = bounds.concretize()

    assert lower.shape == (4, 2, 2)
    assert torch.all(lower <= upper + 1e-5)
    _sample_sound(small_cnn, cnn_region, lower, upper)


@pytest.mark.parametrize("method", PROP_METHODS)
def test_cnn_batched_inputs(
    small_cnn: nn.Module, method: Literal["ibp", "backward_lbp", "forward_lbp", "crown_ibp", "forward_backward_lbp"]
) -> None:
    """Batched verification: feature_shape=(2,8,8), region adds a batch dim."""
    torch.manual_seed(2)
    dummy = torch.zeros(2, 8, 8)
    lower = torch.randn(3, 2, 8, 8)  # batched region → batch_ndim=1
    upper = lower + 0.1
    region = HyperRectangle(lower=lower, upper=upper)

    bm = BoundModel(small_cnn, dummy_inputs=(dummy,), method=method)
    bounds = bm.propagate(region)
    lo, up = bounds.concretize()

    assert lo.shape == (3, 4, 2, 2)
    # Sample per-batch soundness.
    for b in range(3):
        sub_region = HyperRectangle(lower=lower[b], upper=upper[b])
        _sample_sound(small_cnn, sub_region, lo[b], up[b])


@pytest.mark.parametrize("method", PROP_METHODS)
def test_cnn_with_classifier_head(
    method: Literal["ibp", "backward_lbp", "forward_lbp", "crown_ibp", "forward_backward_lbp"],
) -> None:
    """Conv + Pool + flatten-all + Linear on unbatched input."""
    torch.manual_seed(3)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten(start_dim=0)
            self.fc = nn.Linear(3 * 4 * 4, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.conv(x))
            h = self.pool(h)
            h = self.flatten(h)
            return self.fc(h)

    model = Net()
    dummy = torch.zeros(2, 8, 8)
    lower = torch.randn(2, 8, 8)
    upper = lower + 0.1
    region = HyperRectangle(lower=lower, upper=upper)

    bm = BoundModel(model, dummy_inputs=(dummy,), method=method)
    bounds = bm.propagate(region)
    lo, up = bounds.concretize()

    assert lo.shape == (5,)
    assert torch.all(lo <= up + 1e-5)
    _sample_sound(model, region, lo, up)


def test_backward_lbp_tighter_than_ibp() -> None:
    """Backward LBP should be at least as tight as IBP on a small CNN."""
    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
    )
    dummy = torch.zeros(2, 8, 8)
    lower = torch.randn(2, 8, 8)
    upper = lower + 0.2
    region = HyperRectangle(lower=lower, upper=upper)

    ibp_bounds = BoundModel(model, dummy_inputs=(dummy,), method="ibp").propagate(region)
    lbp_bounds = BoundModel(model, dummy_inputs=(dummy,), method="backward_lbp").propagate(region)

    ibp_l, ibp_u = ibp_bounds.concretize()
    lbp_l, lbp_u = lbp_bounds.concretize()

    _sample_sound(model, region, lbp_l, lbp_u)
    assert (lbp_u - lbp_l).mean() <= (ibp_u - ibp_l).mean() + 1e-4


def test_adaptive_avgpool() -> None:
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((2, 2)),
    )
    dummy = torch.zeros(2, 8, 8)
    lower = torch.randn(2, 8, 8)
    upper = lower + 0.1
    region = HyperRectangle(lower=lower, upper=upper)

    for method in PROP_METHODS:
        bm = BoundModel(model, dummy_inputs=(dummy,), method=method)
        bounds = bm.propagate(region)
        lo, up = bounds.concretize()
        assert lo.shape == (2, 2, 2)
        _sample_sound(model, region, lo, up)
