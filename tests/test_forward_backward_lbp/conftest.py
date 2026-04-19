"""Shared test infrastructure for Forward-Backward LBP tests."""

from __future__ import annotations

import torch

from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardBackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.tracer import BoundPropagationTracer


def trace_and_annotate(fn, example_input):
    """Trace *fn* and annotate metadata. Returns the GraphModule."""
    registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn)
    MetadataPass(gm).run(example_input)
    return gm


def propagate_forward_backward(fn, region, example_input=None):
    """Trace *fn*, propagate Forward-Backward LBP, return the first LinearBounds."""
    if example_input is None:
        example_input = torch.randn_like(region.lower)
    gm = trace_and_annotate(fn, example_input)
    propagator = ForwardBackwardLBPPropagator(gm)
    return propagator.propagate([region])[0]


def evaluate_linear_bounds_at(linear_bounds, x):
    """Evaluate affine lower/upper bounds at a concrete point x."""
    output_shape = linear_bounds.bias_lower.shape
    output_ndim = len(output_shape)

    lower = linear_bounds.bias_lower.clone()
    upper = linear_bounds.bias_upper.clone()

    for ll, lu in zip(linear_bounds.linear_lowers, linear_bounds.linear_uppers, strict=True):
        input_ndim = x.ndim
        expanded = x.reshape(*([1] * output_ndim), *x.shape)
        sum_dims = tuple(range(-input_ndim, 0))
        if sum_dims:
            lower = lower + (ll * expanded).sum(dim=sum_dims)
            upper = upper + (lu * expanded).sum(dim=sum_dims)
        else:
            lower = lower + ll * expanded
            upper = upper + lu * expanded

    return lower, upper


def check_soundness(fn, region, linear_bounds, num_samples=1000, atol=1e-4):
    """Sample the region and verify the affine bounds hold pointwise."""
    rand = torch.rand(num_samples, *region.lower.shape)
    samples = region.lower + rand * (region.upper - region.lower)
    for sample in samples:
        output = fn(sample)
        lower, upper = evaluate_linear_bounds_at(linear_bounds, sample)
        if not torch.all(lower <= output + atol):
            raise AssertionError(f"Lower bound violation at x={sample}: lower={lower}, output={output}")
        if not torch.all(output <= upper + atol):
            raise AssertionError(f"Upper bound violation at x={sample}: upper={upper}, output={output}")
