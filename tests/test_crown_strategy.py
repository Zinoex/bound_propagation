"""
Tests for CROWN (forward-mode) bound propagation strategies.

Tests basic functionality of CROWN linear bound propagation.
"""

import pytest
import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.concretize import concretize
from bound_propagation.regions import HyperRectangle
from bound_propagation.strategy import BoundPropagator
from bound_propagation.tracer import GraphConverter, trace_function


class TestCROWNBasic:
    """Test basic CROWN functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Import crown to ensure strategies are registered
        import bound_propagation.strategy.crown  # noqa: F401

    def test_crown_linear_propagation(self):
        """Test CROWN propagation through a simple linear network."""

        # Build a simple linear network: y = ReLU(x @ W + b)
        def model(x):
            weight = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
            bias = torch.tensor([0.5, -0.5])
            return torch.relu(x @ weight + bias)

        fx_module = trace_function(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # Create input region
        lower = torch.tensor([-1.0, -1.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower, upper)

        # Propagate with CROWN
        propagator = BoundPropagator(graph, method="crown")
        bounds_dict = propagator.compute_bounds(region)

        # Get output bounds
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1

        # Output should be LinearBounds
        assert isinstance(output_bounds[0], LinearBounds)

        # Concretize to get intervals
        lower_out, upper_out = concretize(region, output_bounds[0])

        # Verify bounds are valid
        assert lower_out.shape == (2,)
        assert upper_out.shape == (2,)
        assert torch.all(lower_out <= upper_out)

    def test_crown_identity(self):
        """Test CROWN on identity function."""

        def model(x):
            return x

        fx_module = trace_function(model)
        x = torch.zeros(3)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        lower = torch.tensor([0.0, 1.0, 2.0])
        upper = torch.tensor([1.0, 2.0, 3.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="crown")
        bounds_dict = propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1

        lower_out, upper_out = concretize(region, output_bounds[0])

        # For identity, bounds should match input
        assert torch.allclose(lower_out, lower)
        assert torch.allclose(upper_out, upper)

    def test_crown_addition(self):
        """Test CROWN on addition operation."""

        def model(x):
            return x + x

        fx_module = trace_function(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="crown")
        bounds_dict = propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = concretize(region, output_bounds[0])

        # x + x should give [2, 6] and [4, 8]
        assert torch.allclose(lower_out, 2 * lower)
        assert torch.allclose(upper_out, 2 * upper)

    def test_crown_relu_active(self):
        """Test CROWN ReLU when always active."""

        def model(x):
            return torch.relu(x)

        fx_module = trace_function(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # All positive - ReLU is identity
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="crown")
        bounds_dict = propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = concretize(region, output_bounds[0])

        # Should be identity for positive inputs
        assert torch.allclose(lower_out, lower)
        assert torch.allclose(upper_out, upper)

    def test_crown_relu_inactive(self):
        """Test CROWN ReLU when always inactive."""

        def model(x):
            return torch.relu(x)

        fx_module = trace_function(model)
        x = torch.zeros(2)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # All negative - ReLU is zero
        lower = torch.tensor([-3.0, -4.0])
        upper = torch.tensor([-1.0, -2.0])
        region = HyperRectangle(lower, upper)

        propagator = BoundPropagator(graph, method="crown")
        bounds_dict = propagator.compute_bounds(region)

        output_bounds = propagator.get_output_bounds()

        lower_out, upper_out = concretize(region, output_bounds[0])

        # Should be zero for negative inputs
        assert torch.allclose(lower_out, torch.zeros(2))
        assert torch.allclose(upper_out, torch.zeros(2))


class TestCROWNComparison:
    """Compare CROWN with IBP."""

    def setup_method(self):
        """Set up test fixtures."""
        import bound_propagation.strategy.crown  # noqa: F401

    def test_crown_tighter_than_ibp(self):
        """Test that CROWN can give tighter bounds than IBP for ReLU networks."""

        # Simple network with ReLU crossing case
        def model(x):
            # First layer amplifies one dimension
            return torch.relu(x * 2.0)

        fx_module = trace_function(model)
        x = torch.zeros(1)
        converter = GraphConverter(fx_module)
        graph = converter.convert(example_inputs=(x,))

        # Input that crosses zero after first layer
        lower = torch.tensor([-0.5])
        upper = torch.tensor([0.5])
        region = HyperRectangle(lower, upper)

        # Compute with both methods
        propagator_ibp = BoundPropagator(graph, method="ibp")
        propagator_crown = BoundPropagator(graph, method="crown")

        bounds_ibp = propagator_ibp.compute_bounds(region)
        bounds_crown = propagator_crown.compute_bounds(region)

        # Get output bounds
        lower_ibp, upper_ibp = concretize(region, propagator_ibp.get_output_bounds()[0])
        lower_crown, upper_crown = concretize(region, propagator_crown.get_output_bounds()[0])

        # Both should be valid (lower <= upper)
        assert torch.all(lower_ibp <= upper_ibp)
        assert torch.all(lower_crown <= upper_crown)

        # CROWN should give bounds at least as tight as IBP (or tighter)
        assert torch.all(lower_crown >= lower_ibp - 1e-5)
        assert torch.all(upper_crown <= upper_ibp + 1e-5)
