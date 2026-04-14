"""Tests for IBP target-based dispatch and constant handling.

Tests that the TargetRegistry correctly maps fx targets to strategies,
and that strategies correctly handle constant vs abstract inputs.
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.fx as fx

from bound_propagation.bounds import IntervalBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import IBPPropagator, TargetRegistry
from bound_propagation.propagation.ibp import IBPAdd, IBPMatmul, IBPMul, create_default_ibp_registry
from bound_propagation.propagation.registry import normalize_target
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer
from tests.helpers import propagate


class TestTargetRegistryLookup:
    """Test that TargetRegistry lookups and normalize_target work."""

    def test_register_and_get_strategy(self) -> None:
        registry = TargetRegistry()
        strategy = IBPAdd()
        registry.register(torch.add, strategy)

        # Build a minimal graph with a call_function node targeting torch.add
        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        add_node = graph.call_function(torch.add, args=(x, y))
        graph.output(add_node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        assert registry.get_strategy(add_node, gm) is strategy

    def test_lookup_missing_target_raises(self) -> None:
        registry = TargetRegistry()

        graph = fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.relu, args=(x,))
        graph.output(node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with pytest.raises(ValueError, match="No strategy registered"):
            registry.get_strategy(node, gm)

    def test_register_many_targets(self) -> None:
        registry = TargetRegistry()
        strategy = IBPAdd()
        registry.register_many([torch.add, operator.add], strategy)

        assert registry.supports_target(torch.add)
        assert registry.supports_target(operator.add)

    def test_duplicate_registration_raises(self) -> None:
        registry = TargetRegistry()
        registry.register(torch.relu, IBPAdd())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(torch.relu, IBPAdd())

    def test_normalize_target_call_function(self) -> None:
        graph = fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.relu, args=(x,))
        graph.output(node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        assert normalize_target(node, gm) is torch.relu

    def test_normalize_target_call_module(self) -> None:
        mod = torch.nn.Sequential(torch.nn.ReLU())
        graph = fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_module("0", args=(x,))
        graph.output(node)
        gm = fx.GraphModule(mod, graph)

        assert normalize_target(node, gm) is torch.nn.ReLU


class TestIBPConstantHandling:
    """Test that IBP strategies handle constant inputs correctly."""

    def test_add_abstract_and_constant(self) -> None:
        """Test IBPAdd with one abstract and one constant tensor input."""
        bounds = IntervalBounds(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        constant = torch.tensor([2.0, 3.0])

        result = propagate(IBPAdd(), bounds, constant)

        assert isinstance(result, IntervalBounds)
        assert torch.allclose(result.lower, torch.tensor([2.0, 4.0]))
        assert torch.allclose(result.upper, torch.tensor([3.0, 5.0]))

    def test_mul_abstract_and_constant(self) -> None:
        """Test IBPMul with one abstract and one constant tensor input."""
        bounds = IntervalBounds(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        constant = torch.tensor([2.0, -1.0])

        result = propagate(IBPMul(), bounds, constant)

        assert isinstance(result, IntervalBounds)
        assert torch.allclose(result.lower, torch.tensor([0.0, -2.0]))
        assert torch.allclose(result.upper, torch.tensor([2.0, -1.0]))


class TestIBPEndToEndDispatch:
    """Test that IBPPropagator dispatches correctly through a traced graph."""

    def test_add_with_constant_dispatches_correctly(self) -> None:
        """Test that x + constant and x + y use the same IBPAdd strategy."""

        def fn(x):
            return x + torch.tensor([2.0, 3.0])

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2))

        propagator = IBPPropagator(gm, registry)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 1.0]),
                    upper=torch.tensor([1.0, 2.0]),
                )
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([2.0, 4.0]))
        assert torch.allclose(out.upper, torch.tensor([3.0, 5.0]))

    def test_constant_mul_then_add(self) -> None:
        """Test chain: (x + [2, 3]) * [2, -1]."""

        def fn(x):
            return (x + torch.tensor([2.0, 3.0])) * torch.tensor([2.0, -1.0])

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2))

        propagator = IBPPropagator(gm, registry)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 1.0]),
                    upper=torch.tensor([1.0, 2.0]),
                )
            ]
        )

        # x in [0,1]x[1,2], add [2,3] -> [2,3]x[4,5]
        # multiply by [2,-1] -> [4,6]x[-5,-4]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([4.0, -5.0]))
        assert torch.allclose(out.upper, torch.tensor([6.0, -4.0]))

    def test_matmul_with_constant_weight(self) -> None:
        """Test matmul dispatch with constant weight matrix."""

        def fn(x):
            weight = torch.tensor([[1.0, -0.5, 2.0], [0.5, 1.0, -1.0]])
            return x @ weight

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2))

        propagator = IBPPropagator(gm, registry)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 0.0]),
                    upper=torch.tensor([1.0, 1.0]),
                )
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, -0.5, -1.0]))
        assert torch.allclose(out.upper, torch.tensor([1.5, 1.0, 2.0]))

    def test_two_abstract_inputs_subtraction(self) -> None:
        """Test propagation with two abstract inputs (x - y)."""

        def fn(x, y):
            return x - y

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(torch.randn(2), torch.randn(2))

        propagator = IBPPropagator(gm, registry)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([5.0, 5.0]),
                    upper=torch.tensor([10.0, 10.0]),
                ),
                HyperRectangle(
                    lower=torch.tensor([1.0, 2.0]),
                    upper=torch.tensor([1.0, 2.0]),
                ),
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([4.0, 3.0]))
        assert torch.allclose(out.upper, torch.tensor([9.0, 8.0]))
