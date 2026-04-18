"""Shared test infrastructure for backward LBP tests."""

from __future__ import annotations

import torch

from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import BackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def propagate_bound(fn, region, example_input=None):
    """Trace *fn*, propagate backward LBP, return the first LinearBounds.

    Parameters
    ----------
    fn : callable
        A traceable Python function ``x -> y``.
    region : HyperRectangle
        Input region for bound propagation.
    example_input : torch.Tensor | None
        Example input for tracing. If ``None``, uses ``torch.randn_like(region.lower)``.

    Returns
    -------
    LinearBounds
        The first output's linear bounds.
    """
    if example_input is None:
        example_input = torch.randn_like(region.lower)
    registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn)
    MetadataPass(gm).run(example_input)
    propagator = BackwardLBPPropagator(gm)
    outputs = propagator.propagate([region])
    return outputs[0]


def check_soundness(fn, region, lower, upper, num_samples=2000, atol=1e-5):
    """Verify bounds are sound by sampling random points in the region.

    Parameters
    ----------
    fn : callable
        The true function.
    region : HyperRectangle
        Input region.
    lower, upper : torch.Tensor
        Concretized lower and upper bounds.
    num_samples : int
        Number of random samples.
    atol : float
        Absolute tolerance for bound violations.

    Raises
    ------
    AssertionError
        If any sample violates the bounds.
    """
    rand = torch.rand(num_samples, *region.lower.shape)
    samples = region.lower + rand * (region.upper - region.lower)
    for sample in samples:
        output = fn(sample)
        if not torch.all(lower <= output + atol):
            raise AssertionError(
                f"Lower bound violation: lower={lower}, output={output}, "
                f"diff={output - lower}, input={sample}"
            )
        if not torch.all(output <= upper + atol):
            raise AssertionError(
                f"Upper bound violation: upper={upper}, output={output}, "
                f"diff={output - upper}, input={sample}"
            )


def assert_sound(fn, region, example_input=None, num_samples=2000, atol=1e-5):
    """End-to-end soundness: trace, propagate, concretize, sample-check.

    Returns the (lower, upper) tuple for further assertions.
    """
    bounds = propagate_bound(fn, region, example_input)
    lower, upper = bounds.concretize()
    check_soundness(fn, region, lower, upper, num_samples=num_samples, atol=atol)
    return lower, upper


def assert_exact(fn, region, expected_lower, expected_upper, atol=1e-5):
    """Assert concretized bounds match expected values exactly."""
    bounds = propagate_bound(fn, region)
    lower, upper = bounds.concretize()
    assert torch.allclose(lower, expected_lower, atol=atol), (
        f"Lower mismatch: got {lower}, expected {expected_lower}"
    )
    assert torch.allclose(upper, expected_upper, atol=atol), (
        f"Upper mismatch: got {upper}, expected {expected_upper}"
    )
    return bounds


def region(lower, upper):
    """Shorthand for creating a HyperRectangle."""
    return HyperRectangle(
        lower=torch.tensor(lower, dtype=torch.float32),
        upper=torch.tensor(upper, dtype=torch.float32),
    )
