"""Tests for automatic torch.fx wrapping of registered free functions.

When a user registers a custom Python function as a strategy target, the
tracer should keep it as a leaf ``call_function`` node instead of
inlining its body. This is achieved by feeding the registry's function
targets through ``fx.Tracer(autowrap_functions=...)``.
"""

from __future__ import annotations

import torch

from bound_propagation import BoundModel, HyperRectangle, RegistryExtension
from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp import ForwardIBPStrategy, create_default_ibp_registry
from bound_propagation.tracer import BoundPropagationTracer


def scale_then_shift(x: torch.Tensor) -> torch.Tensor:
    """Custom function whose body would be inlined without autowrap."""
    return x * 2.0 + 1.0


class IBPScaleThenShift(ForwardIBPStrategy):
    def propagate_forward(self, node, ctx):
        args, _ = ctx.resolve_args(node)
        x = args[0]
        if not isinstance(x, IntervalBounds):
            raise TypeError("expected IntervalBounds")
        return IntervalBounds(scale_then_shift(x.lower), scale_then_shift(x.upper))


def test_autowrap_keeps_registered_free_function_as_leaf():
    """The registered function is a leaf call_function node, not inlined."""

    def model(x):
        return scale_then_shift(x)

    registry = create_default_ibp_registry()
    registry.register(scale_then_shift, IBPScaleThenShift())

    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(model)

    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert scale_then_shift in targets, f"expected scale_then_shift to be a leaf call_function, got targets={targets}"
    # No inlined mul/add nodes from the body should appear.
    import operator

    assert not any(t in (operator.mul, operator.add, torch.mul, torch.add) for t in targets), (
        f"body of scale_then_shift was inlined; targets={targets}"
    )


def test_autowrap_end_to_end_via_bound_model():
    """A BoundModel built over a custom free function computes correct bounds."""

    def model(x):
        return scale_then_shift(x)

    ext = RegistryExtension(targets=[scale_then_shift], ibp=IBPScaleThenShift())
    bm = BoundModel(model, dummy_inputs=(torch.zeros(4),), method="ibp", extensions=[ext])

    region = HyperRectangle(lower=torch.full((4,), -1.0), upper=torch.full((4,), 1.0))
    bounds = bm.propagate(region)

    # scale_then_shift: y = 2x + 1, so x ∈ [-1, 1] ⇒ y ∈ [-1, 3].
    assert torch.allclose(bounds.lower, torch.full((4,), -1.0))
    assert torch.allclose(bounds.upper, torch.full((4,), 3.0))


def test_autowrap_leaves_native_torch_ops_alone():
    """Passing torch builtins through autowrap_functions must not break tracing."""

    def model(x):
        return torch.relu(x) + 1.0

    # Default registry already contains torch.relu, torch.add etc. —
    # these get fed to autowrap_functions. This should trace cleanly.
    tracer = BoundPropagationTracer(create_default_ibp_registry())
    gm = tracer.trace(model)

    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert torch.relu in targets
