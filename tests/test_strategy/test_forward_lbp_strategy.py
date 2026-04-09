"""
Tests for LBP (forward-mode) bound propagation strategies.

DEPRECATED: Old strategy architecture replaced with method-based propagators.
See tests/test_method_propagators.py for new tests.

Tests basic functionality of LBP linear bound propagation.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Old strategy architecture deprecated")

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.regions import HyperRectangle
# from bound_propagation.strategy import BoundPropagator
from bound_propagation.tracer import BoundPropagationTracer, GraphConverter


def _trace_graph_module(fn_or_module, concrete_args=None):
    tracer = BoundPropagationTracer()
    graph = tracer.trace(fn_or_module, concrete_args=concrete_args)
    return torch.fx.GraphModule(tracer.root, graph)


class TestForwardLBPBasic:
    """Test basic Forward LBP functionality."""

    def test_lbp_linear_propagation(self):
        """Test forward LBP propagation through a simple linear network."""

        # Build a simple linear network: y = ReLU(x @ W + b)
        def model(x):
            weight = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
            bias = torch.tensor([0.5, -0.5])
            return torch.relu(x @ weight + bias)

        fx_module = _trace_graph_module(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # Create input region
        lower = torch.tensor([-1.0, -1.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower, upper)

        # Propagate with forward LBP
        propagator = BoundPropagator(graph, method="forward")
        propagator.compute_bounds(region)

        # Get output bounds
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1

        # Output should be LinearBounds
        assert isinstance(output_bounds[0], LinearBounds)

        # Concretize to get intervals
        lower_out, upper_out = output_bounds[0].concretize()

        # Verify bounds are valid
        assert lower_out.shape == (2,)
        assert upper_out.shape == (2,)
        assert torch.all(lower_out <= upper_out)

    def test_lbp_identity(self):
        """Test LBP on identity function."""

        def model(x):
            return x

        fx_module = _trace_graph_module(model)
        x = torch.zeros(3)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        lower = torch.tensor([0.0, 1.0, 2.0])
        upper = torch.tensor([1.0, 2.0, 3.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="forward")
        propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1

        lower_out, upper_out = output_bounds[0].concretize()

        # For identity, bounds should match input
        assert torch.allclose(lower_out, lower)
        assert torch.allclose(upper_out, upper)

    def test_lbp_addition(self):
        """Test LBP on addition operation."""

        def model(x):
            return x + x

        fx_module = _trace_graph_module(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="forward")
        propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = output_bounds[0].concretize()

        # x + x should give [2, 6] and [4, 8]
        assert torch.allclose(lower_out, 2 * lower)
        assert torch.allclose(upper_out, 2 * upper)

    def test_lbp_relu_active(self):
        """Test LBP ReLU when always active."""

        def model(x):
            return torch.relu(x)

        fx_module = _trace_graph_module(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # All positive - ReLU is identity
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="forward")
        propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = output_bounds[0].concretize()

        # Should be identity for positive inputs
        assert torch.allclose(lower_out, lower)
        assert torch.allclose(upper_out, upper)

    def test_lbp_relu_inactive(self):
        """Test LBP ReLU when always inactive."""

        def model(x):
            return torch.relu(x)

        fx_module = _trace_graph_module(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # All negative - ReLU is zero
        lower = torch.tensor([-3.0, -4.0])
        upper = torch.tensor([-1.0, -2.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="forward")
        propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = output_bounds[0].concretize()

        # Should be zero for negative inputs
        assert torch.allclose(lower_out, torch.zeros(2))
        assert torch.allclose(upper_out, torch.zeros(2))
