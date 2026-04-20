"""Tests for treating torch.zeros/ones/full family as constants under IBP."""

from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import IBPPropagator
from bound_propagation.propagation.ibp import create_default_ibp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_run(fn, example_input: torch.Tensor, region: HyperRectangle) -> IntervalBounds:
    registry = create_default_ibp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn)
    MetadataPass(gm).run(example_input)
    propagator = IBPPropagator(gm)
    outputs = propagator.propagate([region])
    assert isinstance(outputs, IntervalBounds)
    return outputs


def test_zeros_literal_shape() -> None:
    def fn(x):
        return x + torch.zeros(3)

    region = HyperRectangle(lower=torch.tensor([1.0, 2.0, 3.0]), upper=torch.tensor([2.0, 3.0, 4.0]))
    out = _trace_and_run(fn, torch.randn(3), region)
    assert torch.allclose(out.lower, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 3.0, 4.0]))


def test_ones_literal_shape() -> None:
    def fn(x):
        return x + torch.ones(3)

    region = HyperRectangle(lower=torch.tensor([0.0, 0.0, 0.0]), upper=torch.tensor([1.0, 1.0, 1.0]))
    out = _trace_and_run(fn, torch.randn(3), region)
    assert torch.allclose(out.lower, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 2.0, 2.0]))


def test_full_literal_shape() -> None:
    def fn(x):
        return x * torch.full((3,), 2.0)

    region = HyperRectangle(lower=torch.tensor([1.0, 1.0, 1.0]), upper=torch.tensor([2.0, 2.0, 2.0]))
    out = _trace_and_run(fn, torch.randn(3), region)
    assert torch.allclose(out.lower, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0, 4.0, 4.0]))


def test_zeros_like_abstract_input() -> None:
    def fn(x):
        return x + torch.zeros_like(x)

    region = HyperRectangle(lower=torch.tensor([1.0, 2.0, 3.0]), upper=torch.tensor([2.0, 3.0, 4.0]))
    out = _trace_and_run(fn, torch.randn(3), region)
    assert torch.allclose(out.lower, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 3.0, 4.0]))


def test_ones_like_abstract_input() -> None:
    def fn(x):
        return x * torch.ones_like(x)

    region = HyperRectangle(lower=torch.tensor([-1.0, 2.0]), upper=torch.tensor([1.0, 3.0]))
    out = _trace_and_run(fn, torch.randn(2), region)
    assert torch.allclose(out.lower, torch.tensor([-1.0, 2.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0, 3.0]))


def test_full_like_abstract_input() -> None:
    def fn(x):
        return x + torch.full_like(x, 5.0)

    region = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([1.0, 2.0]))
    out = _trace_and_run(fn, torch.randn(2), region)
    assert torch.allclose(out.lower, torch.tensor([5.0, 5.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0, 7.0]))
