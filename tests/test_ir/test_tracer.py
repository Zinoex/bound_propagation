"""
Tests for graph tracing and conversion.
"""

import pytest
import torch
import torch.nn as nn

from bound_propagation.ir import Graph, NodeType, OperationType, TensorMetadata
from bound_propagation.tracer import BoundPropagationTracer, GraphConverter
from bound_propagation.tracer.fx_tracer import TraceError, UnsupportedOperationError


def _trace_graph_module(fn_or_module, concrete_args=None):
    tracer = BoundPropagationTracer()
    graph = tracer.trace(fn_or_module, concrete_args=concrete_args)
    return torch.fx.GraphModule(tracer.root, graph)


class TestTraceFunction:
    """Tests for torch.fx tracing."""

    def test_trace_simple_function(self):
        """Test tracing a simple function."""

        def simple_fn(x):
            return torch.relu(x)

        traced = _trace_graph_module(simple_fn)
        assert traced is not None
        assert isinstance(traced, torch.fx.GraphModule)

    def test_trace_function_with_multiple_ops(self):
        """Test tracing function with multiple operations."""

        def multi_op_fn(x, y):
            z = x + y
            return torch.relu(z)

        traced = _trace_graph_module(multi_op_fn)
        assert traced is not None

        # Count operations in graph
        num_ops = sum(1 for node in traced.graph.nodes if node.op in ["call_function", "call_method"])
        assert num_ops >= 2  # At least add and relu

    def test_trace_with_example_inputs(self):
        """Test tracing with example inputs for validation."""

        def fn(x):
            return x * 2 + 1

        x = torch.randn(3, 4)
        traced = _trace_graph_module(fn)

        # Verify outputs match
        original_out = fn(x)
        traced_out = traced(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_nn_module(self):
        """Test tracing a torch.nn.Module."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return torch.relu(self.linear(x))

        module = SimpleModule()
        traced = _trace_graph_module(module)
        assert traced is not None

        # Test with example input
        x = torch.randn(2, 10)
        original_out = module(x)
        traced_out = traced(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_with_residual_connection(self):
        """Test tracing function with residual connection."""

        def residual_fn(x):
            identity = x
            x = torch.relu(x)
            return x + identity

        traced = _trace_graph_module(residual_fn)
        assert traced is not None

        x = torch.randn(3, 4)
        original_out = residual_fn(x)
        traced_out = traced(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_control_flow_fails(self):
        """Test that dynamic control flow fails under torch.fx tracing."""

        def fn_with_if(x):
            if x.sum() > 0:
                return torch.relu(x)
            return x

        with pytest.raises((TraceError, RuntimeError, torch.fx.proxy.TraceError)):
            _trace_graph_module(fn_with_if)

    def test_trace_unsupported_operation_rejected(self):
        """Test that tracing rejects operations not present in the mapping."""

        def unsupported_fn(x):
            return torch.erf(x)

        with pytest.raises(UnsupportedOperationError):
            _trace_graph_module(unsupported_fn)


class TestGraphConverter:
    """Tests for fx.GraphModule to IR Graph conversion."""

    def test_convert_simple_graph(self):
        """Test converting a simple traced graph to IR."""

        def simple_fn(x):
            return torch.relu(x)

        # Trace and convert
        fx_module = _trace_graph_module(simple_fn)
        x = torch.randn(3, 4)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        assert isinstance(ir_graph, Graph)
        assert ir_graph.num_nodes > 0
        assert ir_graph.num_inputs > 0
        assert ir_graph.num_outputs > 0

    def test_convert_preserves_structure(self):
        """Test that conversion preserves graph structure."""

        def fn(x, y):
            z = x + y
            return torch.relu(z)

        fx_module = _trace_graph_module(fn)
        x, y = torch.randn(3, 4), torch.randn(3, 4)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x, y))

        # Should have 2 inputs
        assert ir_graph.num_inputs == 2

        # Should have nodes for inputs, add, relu
        assert ir_graph.num_nodes >= 4  # 2 inputs + add + relu

        # Check operation types are correct
        op_types = [node.op_type for node in ir_graph.nodes]
        assert OperationType.INPUT in op_types
        assert OperationType.ADD in op_types
        assert OperationType.RELU in op_types

    def test_convert_extracts_metadata(self):
        """Test that metadata (shape, dtype) is extracted correctly."""

        def fn(x):
            return x * 2

        x = torch.randn(3, 4, dtype=torch.float32)
        fx_module = _trace_graph_module(fn)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        # Check input node has correct metadata
        input_nodes = ir_graph.input_nodes
        assert len(input_nodes) > 0

        input_node = input_nodes[0]
        assert isinstance(input_node.output_metadata, TensorMetadata)
        assert input_node.output_metadata.shape == (3, 4)
        assert "float32" in input_node.output_metadata.dtype

    def test_convert_linear_layer(self):
        """Test converting a linear layer."""

        class LinearModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        module = LinearModule()
        fx_module = _trace_graph_module(module)
        x = torch.randn(2, 10)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        # Should have linear operation
        op_types = [node.op_type for node in ir_graph.nodes]
        assert OperationType.LINEAR in op_types or OperationType.MATMUL in op_types

    def test_convert_multi_input_operation(self):
        """Test converting operation with multiple inputs."""

        def fn(x, y):
            return x @ y  # matmul

        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        fx_module = _trace_graph_module(fn)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x, y))

        # Find matmul node
        matmul_nodes = [node for node in ir_graph.nodes if node.op_type == OperationType.MATMUL]
        assert len(matmul_nodes) > 0

        matmul_node = matmul_nodes[0]
        assert matmul_node.num_inputs == 2

    def test_convert_validates_graph(self):
        """Test that converter validates the resulting graph."""

        def fn(x):
            return torch.relu(x)

        fx_module = _trace_graph_module(fn)
        x = torch.randn(3, 4)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        # Graph should be valid (no exception raised)
        ir_graph.validate()

    def test_convert_topological_order(self):
        """Test that converted graph maintains topological order."""

        def fn(x):
            a = x + 1
            b = torch.relu(a)
            c = b * 2
            return c

        fx_module = _trace_graph_module(fn)
        x = torch.randn(3, 4)

        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        # Get topological order
        topo_order = ir_graph.topological_order()

        # Input should come first
        assert topo_order[0].node_type == NodeType.INPUT

        # All nodes should be in valid topological order
        # (each node appears after its dependencies)
        seen = set()
        for node in topo_order:
            for input_node in node.inputs:
                assert input_node in seen, f"Node {node.name} appears before its input {input_node.name}"
            seen.add(node)


class TestEndToEnd:
    """End-to-end tests for tracing and conversion."""

    def test_trace_and_convert_simple(self):
        """Test complete pipeline: trace -> convert."""

        def fn(x):
            return torch.relu(x + 1)

        # Trace
        fx_module = _trace_graph_module(fn)

        # Convert
        x = torch.randn(3, 4)
        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x,))

        # Verify
        assert ir_graph.num_inputs == 1
        assert ir_graph.num_outputs == 1
        assert ir_graph.num_nodes >= 3  # input, add, relu

    def test_trace_and_convert_complex(self):
        """Test pipeline with more complex function."""

        def complex_fn(x, y):
            z1 = x @ y
            z2 = torch.relu(z1)
            z3 = z2 + z1  # Add z1 instead of x (same shape)
            return torch.tanh(z3)

        # Trace
        fx_module = _trace_graph_module(complex_fn)

        # Convert
        x = torch.randn(4, 5)
        y = torch.randn(5, 4)
        converter = GraphConverter(fx_module)
        ir_graph = converter.convert(example_inputs=(x, y))

        # Verify
        assert ir_graph.num_inputs == 2
        assert ir_graph.num_outputs == 1

        # Check all expected operations present
        op_types = [node.op_type for node in ir_graph.nodes]
        assert OperationType.MATMUL in op_types
        assert OperationType.RELU in op_types
        assert OperationType.ADD in op_types
        assert OperationType.TANH in op_types
