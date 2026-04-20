"""
Tests for graph tracing with BoundPropagationTracer.
"""

import pytest
import torch
import torch.nn as nn

from bound_propagation.propagation.ibp import create_default_ibp_registry
from bound_propagation.tracer import BoundPropagationTracer
from bound_propagation.tracer.fx_tracer import MultiOutputError, TraceError, UnsupportedOperationError


def _default_registry():
    return create_default_ibp_registry()


def _trace(fn_or_module, registry=None):
    if registry is None:
        registry = _default_registry()
    tracer = BoundPropagationTracer(registry)
    return tracer.trace(fn_or_module)


class TestTraceFunction:
    """Tests for torch.fx tracing with registry validation."""

    def test_trace_simple_function(self):
        def simple_fn(x):
            return torch.relu(x)

        gm = _trace(simple_fn)
        assert gm is not None
        assert isinstance(gm, torch.fx.GraphModule)

    def test_trace_function_with_multiple_ops(self):
        def multi_op_fn(x, y):
            z = x + y
            return torch.relu(z)

        gm = _trace(multi_op_fn)
        num_ops = sum(1 for n in gm.graph.nodes if n.op in ("call_function", "call_method"))
        assert num_ops >= 2

    def test_trace_preserves_semantics(self):
        def fn(x):
            return x * 2 + 1

        x = torch.randn(3, 4)

        # Trace with registry that explicitly supports the needed ops
        gm = _trace(fn)

        original_out = fn(x)
        traced_out = gm(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_nn_module(self):
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return torch.relu(self.linear(x))

        module = SimpleModule()
        gm = _trace(module)
        assert gm is not None

        x = torch.randn(2, 10)
        original_out = module(x)
        traced_out = gm(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_with_residual_connection(self):
        def residual_fn(x):
            identity = x
            x = torch.relu(x)
            return x + identity

        gm = _trace(residual_fn)
        x = torch.randn(3, 4)
        original_out = residual_fn(x)
        traced_out = gm(x)
        assert torch.allclose(original_out, traced_out)

    def test_trace_control_flow_fails(self):
        def fn_with_if(x):
            if x.sum() > 0:
                return torch.relu(x)
            return x

        with pytest.raises((TraceError, RuntimeError, torch.fx.proxy.TraceError)):
            _trace(fn_with_if)

    def test_trace_unsupported_operation_rejected(self):
        def unsupported_fn(x):
            return torch.erf(x)

        with pytest.raises(UnsupportedOperationError):
            _trace(unsupported_fn)

    def test_trace_validates_all_operations(self):
        """Test that unsupported operations in a multi-op graph are caught."""

        def mixed_fn(x):
            y = torch.relu(x)  # supported
            return torch.erf(y)  # unsupported

        with pytest.raises(UnsupportedOperationError):
            _trace(mixed_fn)


class TestTracerWithModules:
    """Test tracer behavior with nn.Module types."""

    def test_registered_module_kept_as_leaf(self):
        """Registered module types should be kept as leaf (call_module) nodes."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        gm = _trace(Model())

        # Should have a call_module node for the linear layer
        module_nodes = [n for n in gm.graph.nodes if n.op == "call_module"]
        assert len(module_nodes) >= 1

    def test_unregistered_module_is_traced_into(self):
        """Unregistered module types should be traced into (not call_module)."""

        class CustomLayer(nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom = CustomLayer()

            def forward(self, x):
                return self.custom(x)

        gm = _trace(Model())

        # CustomLayer is not registered, so its internals should be expanded
        module_nodes = [n for n in gm.graph.nodes if n.op == "call_module"]
        # Should not have a call_module for "custom"
        module_targets = [n.target for n in module_nodes]
        assert "custom" not in module_targets


class TestEndToEnd:
    """End-to-end tests for tracing."""

    def test_trace_simple_pipeline(self):
        def fn(x):
            return torch.relu(x + 1)

        gm = _trace(fn)

        # Should have placeholders + ops + output
        node_ops = [n.op for n in gm.graph.nodes]
        assert "placeholder" in node_ops
        assert "output" in node_ops
        call_ops = [n for n in gm.graph.nodes if n.op.startswith("call_")]
        assert len(call_ops) >= 2  # add and relu

    def test_trace_complex_pipeline(self):
        def complex_fn(x, y):
            z1 = torch.matmul(x, y)
            z2 = torch.relu(z1)
            z3 = z2 + z1
            return torch.tanh(z3)

        gm = _trace(complex_fn)

        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

        x = torch.randn(4, 5)
        y = torch.randn(5, 4)
        original_out = complex_fn(x, y)
        traced_out = gm(x, y)
        assert torch.allclose(original_out, traced_out)


class TestTraceMultiOutputRejection:
    """Bound propagation supports single-output functions only."""

    def test_tuple_return_raises(self):
        def fn(x):
            return torch.relu(x), torch.tanh(x)

        with pytest.raises(MultiOutputError, match="returns 2 values"):
            _trace(fn)

    def test_list_return_raises(self):
        def fn(x):
            return [torch.relu(x), x + 1.0]

        with pytest.raises(MultiOutputError):
            _trace(fn)

    def test_single_output_still_traces(self):
        def fn(x):
            return torch.relu(x)

        _trace(fn)  # no raise
