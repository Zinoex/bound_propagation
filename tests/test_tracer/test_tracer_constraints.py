"""
Tests for tracer constraints.

The tracer accepts any operation registered in the TargetRegistry.
Operations not in the registry are rejected at trace time.
"""

import pytest
import torch
import torch.nn as nn

from bound_propagation.propagation.ibp import create_default_ibp_registry
from bound_propagation.tracer import BoundPropagationTracer
from bound_propagation.tracer.fx_tracer import UnsupportedOperationError


def _default_registry():
    return create_default_ibp_registry()


def _trace(fn_or_module, registry=None):
    if registry is None:
        registry = _default_registry()
    tracer = BoundPropagationTracer(registry)
    return tracer.trace(fn_or_module)


class TestTracerConstraints:
    """Test that tracer accepts/rejects operations based on registry."""

    def test_trace_standard_function(self):
        def model(x):
            return torch.relu(x * 2.0 + 1.0)

        gm = _trace(model)
        assert gm is not None

    def test_trace_complex_function(self):
        def model(x):
            h1 = torch.relu(x)
            h2 = torch.sigmoid(h1)
            h3 = torch.tanh(h2)
            return h3 * 2.0 - 1.0

        gm = _trace(model)
        assert gm is not None

    def test_discrete_indexing_is_allowed(self):
        """operator.getitem is registered, so indexing works."""

        def model(x):
            return x[:, 0]

        gm = _trace(model)
        assert gm is not None

    def test_unmapped_function_rejected(self):
        def model(x):
            return torch.erf(x)

        with pytest.raises(UnsupportedOperationError):
            _trace(model)

    def test_nested_unmapped_module_rejected(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(x)

        with pytest.raises(UnsupportedOperationError):
            _trace(Model())

    def test_registered_activations_accepted(self):
        """All registered activations should trace without error."""
        activations = [
            lambda x: torch.relu(x),
            lambda x: torch.sigmoid(x),
            lambda x: torch.tanh(x),
            lambda x: torch.exp(x),
            lambda x: torch.log(torch.abs(x) + 1.0),
            lambda x: torch.sqrt(torch.abs(x) + 1.0),
            lambda x: torch.abs(x),
            lambda x: torch.sin(x),
            lambda x: torch.cos(x),
            lambda x: torch.tan(x * 0.1),
            lambda x: torch.neg(x),
            lambda x: torch.reciprocal(torch.abs(x) + 1.0),
        ]
        for fn in activations:
            gm = _trace(fn)
            assert gm is not None

    def test_registered_reductions_accepted(self):
        """Registered reduction operations should trace."""
        reductions = [
            lambda x: torch.sum(x, dim=1),
            lambda x: torch.mean(x, dim=1),
            lambda x: torch.amax(x, dim=1),
            lambda x: torch.amin(x, dim=1),
        ]
        for fn in reductions:
            gm = _trace(fn)
            assert gm is not None

    def test_clamp_accepted(self):
        def model(x):
            return torch.clamp(x, min=-0.5, max=0.5)

        gm = _trace(model)
        assert gm is not None

    def test_matmul_accepted(self):
        def model(x, y):
            return torch.matmul(x, y)

        gm = _trace(model)
        assert gm is not None

    def test_cat_accepted(self):
        def model(x):
            return torch.cat([x, x], dim=1)

        gm = _trace(model)
        assert gm is not None
