"""Shared helpers for alpha-CROWN propagator tests."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from bound_propagation.passes import MetadataPass
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def trace_fn(fn: Callable, example_input: torch.Tensor, registry):
    """Trace ``fn`` with ``registry`` and attach metadata using ``example_input``."""
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn)
    MetadataPass(gm).run(example_input)
    return gm


def region(lower: list[float], upper: list[float]) -> HyperRectangle:
    return HyperRectangle(
        lower=torch.tensor(lower, dtype=torch.float32),
        upper=torch.tensor(upper, dtype=torch.float32),
    )


def check_sound_vs_samples(
    fn: Callable,
    region_: HyperRectangle,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int = 2000,
    atol: float = 1e-4,
) -> None:
    """Verify bounds are sound by sampling uniformly in the region."""
    rand = torch.rand(num_samples, *region_.lower.shape)
    samples = region_.lower + rand * (region_.upper - region_.lower)
    for sample in samples:
        output = fn(sample)
        if not torch.all(lower <= output + atol):
            raise AssertionError(
                f"Lower bound violation: lower={lower.tolist()}, output={output.tolist()}, "
                f"diff={(output - lower).tolist()}, input={sample.tolist()}"
            )
        if not torch.all(output <= upper + atol):
            raise AssertionError(
                f"Upper bound violation: upper={upper.tolist()}, output={output.tolist()}, "
                f"diff={(output - upper).tolist()}, input={sample.tolist()}"
            )


def bound_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    """Sum of (upper - lower) across every output."""
    return float((upper - lower).sum().item())


def make_relu_net(input_dim: int = 3, hidden: int = 8, output_dim: int = 2, seed: int = 0) -> nn.Module:
    """Small deterministic ReLU MLP for alpha-CROWN smoke tests."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )
