"""Tests for Forward LBP target-based dispatch and constant handling.

Tests that the TargetRegistry correctly maps fx targets to strategies for
Forward LBP, and that strategies correctly handle constant vs abstract inputs.
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardLBPPropagator, TargetRegistry
from bound_propagation.propagation.forward_lbp import (
    ForwardLBPAdd,
    ForwardLBPMatmul,
    ForwardLBPMul,
    create_default_forward_lbp_registry,
)
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer
from tests.helpers import propagate


class TestForwardLBPConstantHandling:
    """Test that Forward LBP strategies handle constant inputs correctly."""

    def test_add_abstract_and_constant(self) -> None:
        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = LinearBounds(
            regions=[region],
            linear_lower=torch.eye(2),
            bias_lower=torch.zeros(2),
            linear_upper=torch.eye(2),
            bias_upper=torch.zeros(2),
        )
        constant = torch.tensor([2.0, 3.0])

        result = propagate(ForwardLBPAdd(), bounds, constant)

        assert isinstance(result, LinearBounds)
        lower, upper = result.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, 3.0]))
        assert torch.allclose(upper, torch.tensor([3.0, 4.0]))

    def test_mul_abstract_and_constant(self) -> None:
        region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([3.0, 3.0]),
        )
        bounds = LinearBounds(
            regions=[region],
            linear_lower=torch.eye(2),
            bias_lower=torch.zeros(2),
            linear_upper=torch.eye(2),
            bias_upper=torch.zeros(2),
        )
        constant = torch.tensor([2.0, -1.0])

        result = propagate(ForwardLBPMul(), bounds, constant)

        assert isinstance(result, LinearBounds)
        lower, upper = result.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, -3.0]))
        assert torch.allclose(upper, torch.tensor([6.0, -1.0]))


class TestForwardLBPEndToEndDispatch:
    """Test that ForwardLBPPropagator dispatches correctly through a traced graph."""

    def test_add_with_constant(self) -> None:
        def fn(x):
            return x + torch.tensor([2.0, 2.0])

        registry = create_default_forward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2))

        propagator = ForwardLBPPropagator(gm, registry)
        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        result = outputs[0]
        assert isinstance(result, LinearBounds)
        lower, upper = result.concretize()
        assert torch.allclose(lower, torch.tensor([3.0, 3.0]))
        assert torch.allclose(upper, torch.tensor([4.0, 4.0]))

    def test_computation_with_mixed_signatures(self) -> None:
        """Test: x + 2 -> mul 3: 3(x + 2) for x in [1, 2]."""

        def fn(x):
            return (x + torch.tensor([2.0, 2.0])) * torch.tensor([3.0, 3.0])

        registry = create_default_forward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2))

        propagator = ForwardLBPPropagator(gm, registry)
        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        result = outputs[0]
        assert isinstance(result, LinearBounds)
        lower, upper = result.concretize()
        assert torch.allclose(lower, torch.tensor([9.0, 9.0]))
        assert torch.allclose(upper, torch.tensor([12.0, 12.0]))

    def test_matmul_with_constant_weight(self) -> None:
        """Test Forward LBP dispatch for matmul with constant weight."""

        def fn(x):
            weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            bias = torch.tensor([1.0, -1.0])
            return x @ weight + bias

        registry = create_default_forward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(1, 3))

        propagator = ForwardLBPPropagator(gm, registry)
        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0, 1.0]]),
        )
        outputs = propagator.propagate([input_region])

        result = outputs[0]
        assert isinstance(result, LinearBounds)
        lower, upper = result.concretize()
        # x @ W: [0,1]^3 @ [[1,2],[3,4],[5,6]] -> col0: [0,9], col1: [0,12]
        # + bias: [1, 10], [-1, 11]
        assert torch.allclose(lower, torch.tensor([[1.0, -1.0]]))
        assert torch.allclose(upper, torch.tensor([[10.0, 11.0]]))
