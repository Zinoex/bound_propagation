"""
Tests for tracer constraints.

The tracer accepts any operation that can be represented in our IR mapping.
This intentionally includes piecewise, discrete-indexing, and non-smooth ops
that still participate in PyTorch autograd.
"""

import pytest
import torch
import torch.nn as nn

from bound_propagation.ir.operations import OperationType
from bound_propagation.tracer import BoundPropagationTracer
from bound_propagation.tracer.fx_tracer import UnsupportedOperationError
from bound_propagation.tracer.op_mapping import get_operation_type


def _trace_graph_module(fn_or_module, concrete_args=None):
    tracer = BoundPropagationTracer()
    graph = tracer.trace(fn_or_module, concrete_args=concrete_args)
    return torch.fx.GraphModule(tracer.root, graph)


class TestOperationProperties:
    """Test operation classification properties."""

    def test_linear_ops_properties(self):
        """Test that linear operations have correct properties."""
        assert OperationType.MATMUL.is_linear
        assert OperationType.LINEAR.is_linear
        assert OperationType.TRANSPOSE.is_linear

    def test_activation_ops_properties(self):
        """Test that activation operations have correct properties."""
        assert OperationType.RELU.is_activation
        assert OperationType.SIGMOID.is_activation
        assert OperationType.TANH.is_activation
        assert OperationType.EXP.is_activation

    def test_reduction_ops_properties(self):
        """Test that reduction operations have correct properties."""
        assert OperationType.SUM.is_reduction
        assert OperationType.MEAN.is_reduction
        assert OperationType.MAX.is_reduction
        assert OperationType.MIN.is_reduction

    def test_structural_ops_properties(self):
        """Test that structural operations have correct properties."""
        assert OperationType.RESHAPE.is_structural
        assert OperationType.FLATTEN.is_structural
        assert OperationType.PERMUTE.is_structural
        assert OperationType.CONCAT.is_structural


class TestTracerConstraints:
    """Test that tracer accepts autograd-capable operation patterns."""

    def test_trace_standard_function(self):
        """Test tracing a function with common smooth/non-smooth ops."""

        def model(x):
            return torch.relu(x * 2.0 + 1.0)

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_trace_complex_function(self):
        """Test tracing a complex function with multiple operations."""

        def model(x):
            h1 = torch.relu(x)
            h2 = torch.sigmoid(h1)
            h3 = torch.tanh(h2)
            return h3 * 2.0 - 1.0

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_max_is_allowed(self):
        """Test that max reductions are accepted by the tracer mapping."""

        def model(x):
            values, _ = torch.max(x, dim=1)
            return values

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_min_is_allowed(self):
        """Test that min reductions are accepted by the tracer mapping."""

        def model(x):
            values, _ = torch.min(x, dim=1)
            return values

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_discrete_indexing_is_allowed(self):
        """Test that discrete indexing patterns are accepted."""

        def model(x):
            return x[:, 0]

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_gather_is_allowed(self):
        """Test that gather is accepted."""

        def model(x, idx):
            return torch.gather(x, dim=1, index=idx)

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_heaviside_is_allowed(self):
        """Test that heaviside is accepted by the op mapping."""

        def model(x):
            return torch.heaviside(x, values=torch.tensor(0.0))

        traced = _trace_graph_module(model)
        assert traced is not None

    def test_unmapped_function_rejected(self):
        """Test that unmapped torch functions are rejected at trace time."""

        def model(x):
            return torch.erf(x)

        with pytest.raises(UnsupportedOperationError):
            _trace_graph_module(model)

    def test_nested_unmapped_module_rejected(self):
        """Test that nested unmapped modules are traced into then rejected."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(x)

        with pytest.raises(UnsupportedOperationError):
            _trace_graph_module(Model())


class TestOperationMapping:
    """Test explicit op mapping for requested operations."""

    def test_mapping_max_min(self):
        assert get_operation_type(torch.max) == OperationType.MAX
        assert get_operation_type(torch.min) == OperationType.MIN

    def test_mapping_gather(self):
        assert get_operation_type(torch.gather) == OperationType.GATHER

    def test_mapping_heaviside(self):
        # Heaviside is represented as a clamp-like piecewise op in the current IR.
        assert get_operation_type(torch.heaviside) == OperationType.CLAMP
