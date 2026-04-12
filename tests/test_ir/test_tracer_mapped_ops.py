"""High-coverage tests for tracing mapped operations.

These tests exercise a broad set of mapped torch operations (function, method,
and module forms) in isolation and in compositions, using BoundPropagationTracer
explicitly.
"""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.tracer import BoundPropagationTracer


def _trace_graph_module(fn_or_module, concrete_args=None):
    tracer = BoundPropagationTracer()
    graph = tracer.trace(fn_or_module, concrete_args=concrete_args)
    return torch.fx.GraphModule(tracer.root, graph)


def _assert_outputs_close(original, traced) -> None:
    if isinstance(original, torch.Tensor):
        assert isinstance(traced, torch.Tensor)
        if original.dtype.is_floating_point:
            assert torch.allclose(original, traced, rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(original, traced)
        return

    if isinstance(original, (tuple, list)):
        assert isinstance(traced, (tuple, list))
        assert len(original) == len(traced)
        for o, t in zip(original, traced, strict=True):
            _assert_outputs_close(o, t)
        return

    assert original == traced


def _run_trace_equivalence(fn, *args):
    traced = _trace_graph_module(fn)
    original_out = fn(*args)
    traced_out = traced(*args)
    _assert_outputs_close(original_out, traced_out)


def _collect_trace_targets(
    traced: torch.fx.GraphModule,
) -> tuple[set[object], set[str], set[type[torch.nn.Module]]]:
    function_targets: set[object] = set()
    method_targets: set[str] = set()
    module_types: set[type[torch.nn.Module]] = set()

    for node in traced.graph.nodes:
        if node.op == "call_function":
            function_targets.add(node.target)
        elif node.op == "call_method":
            method_targets.add(str(node.target))
        elif node.op == "call_module":
            module = traced.get_submodule(str(node.target))
            module_types.add(type(module))

    return function_targets, method_targets, module_types


def _assert_any_present(options: set[object], seen: set[object]) -> None:
    assert any(opt in seen for opt in options), f"None of {options} found in seen targets"


def _assert_function_or_method_present(
    function_options: set[object],
    method_options: set[str],
    seen_functions: set[object],
    seen_methods: set[str],
) -> None:
    has_function = any(opt in seen_functions for opt in function_options)
    has_method = any(opt in seen_methods for opt in method_options)
    assert has_function or has_method


def test_trace_mapped_function_ops_equivalence():
    x2 = torch.randn(4, 5)
    y2 = torch.randn(5, 3)
    x3 = torch.randn(2, 3, 4)
    y3 = torch.randn(2, 4, 5)
    gather_idx = torch.tensor([[0, 1], [2, 3], [1, 0], [4, 2]], dtype=torch.long)
    heaviside_values = torch.tensor(0.0)

    cases = [
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
        lambda x: torch.clamp(x, min=-0.5, max=0.5),
        lambda x: torch.heaviside(x, values=heaviside_values),
        lambda x: torch.neg(x),
        lambda x: torch.reciprocal(torch.abs(x) + 1.0),
        lambda x: torch.pow(x, 2.0),
        lambda x: torch.sum(x, dim=1),
        lambda x: torch.mean(x, dim=1),
        lambda x: torch.max(x, dim=1)[0],
        lambda x: torch.min(x, dim=1)[0],
        lambda x: torch.reshape(x, (2, 10)),
        lambda x: torch.flatten(x),
        lambda x: torch.unsqueeze(x, dim=0),
        lambda x: torch.squeeze(torch.unsqueeze(x, dim=0), dim=0),
        lambda x: torch.transpose(torch.reshape(x, (2, 2, 5)), 1, 2),
        lambda x: torch.permute(torch.reshape(x, (2, 2, 5)), (0, 2, 1)),
        lambda x: torch.gather(x, dim=1, index=gather_idx),
        lambda x: x[:, 1:4],
    ]

    for fn in cases:
        _run_trace_equivalence(fn, x2)

    _run_trace_equivalence(lambda a, b: torch.matmul(a, b), x2, y2)
    _run_trace_equivalence(lambda a, b: torch.mm(a, b), x2, y2)
    _run_trace_equivalence(lambda a, b: torch.bmm(a, b), x3, y3)

    _run_trace_equivalence(lambda x: torch.cat([x, x], dim=1), x2)
    _run_trace_equivalence(lambda x: torch.concat([x, x], dim=1), x2)
    _run_trace_equivalence(lambda x: torch.split(x, 2, dim=1), x2)

    _run_trace_equivalence(lambda x: x + 2.0, x2)
    _run_trace_equivalence(lambda x: x - 2.0, x2)
    _run_trace_equivalence(lambda x: x * 2.0, x2)
    _run_trace_equivalence(lambda x: x / 2.0, x2)


def test_trace_mapped_method_ops_equivalence():
    x = torch.randn(4, 5)

    _run_trace_equivalence(lambda t: t.relu(), x)
    _run_trace_equivalence(lambda t: t.sigmoid(), x)
    _run_trace_equivalence(lambda t: t.tanh(), x)
    _run_trace_equivalence(lambda t: t.exp().log(), x)
    _run_trace_equivalence(lambda t: t.abs().sqrt(), x)
    _run_trace_equivalence(lambda t: t.sin() + t.cos() + t.tan() * 0.0, x)
    _run_trace_equivalence(lambda t: t.reshape(2, 10).flatten(), x)
    _run_trace_equivalence(lambda t: t.reshape(2, 2, 5).transpose(1, 2), x)
    _run_trace_equivalence(lambda t: t.reshape(2, 2, 5).permute(0, 2, 1), x)
    _run_trace_equivalence(lambda t: t.sum(dim=1), x)
    _run_trace_equivalence(lambda t: t.mean(dim=1), x)
    _run_trace_equivalence(lambda t: t.max(dim=1)[0], x)
    _run_trace_equivalence(lambda t: t.min(dim=1)[0], x)


def test_trace_mapped_modules_equivalence():
    module_cases = [
        (nn.Linear(5, 3), torch.randn(2, 5)),
        (nn.ReLU(), torch.randn(2, 5)),
        (nn.Sigmoid(), torch.randn(2, 5)),
        (nn.Tanh(), torch.randn(2, 5)),
        (nn.Flatten(), torch.randn(2, 3, 4)),
    ]

    for module, x in module_cases:
        _run_trace_equivalence(module, x)


def test_trace_functional_linear_equivalence():
    x = torch.randn(3, 5)
    w = torch.randn(4, 5)
    b = torch.randn(4)

    _run_trace_equivalence(lambda a, wt, bt: F.linear(a, wt, bt), x, w, b)


def test_trace_composed_mapped_pipeline_equivalence():
    def composed(x, y, idx):
        z = torch.matmul(x, y)
        z = torch.relu(z)
        z = z + torch.sigmoid(z)
        z = torch.tanh(z) * 0.5
        z = torch.clamp(z, min=-1.0, max=1.0)
        z = torch.sin(z) + torch.cos(z)
        z = torch.sqrt(torch.abs(z) + 1e-2)
        z = torch.reshape(z, (2, 6))
        z = torch.unsqueeze(z, dim=0)
        z = torch.squeeze(z, dim=0)
        z = torch.gather(z, dim=1, index=idx)
        left, right = torch.split(z, [2, 2], dim=1)
        out = torch.concat([left, right], dim=1)
        return out

    x = torch.randn(2, 4)
    y = torch.randn(4, 6)
    idx = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)

    traced = _trace_graph_module(composed)
    original_out = composed(x, y, idx)
    traced_out = traced(x, y, idx)

    _assert_outputs_close(original_out, traced_out)

    traced_ops = [node.op for node in traced.graph.nodes if node.op.startswith("call_")]
    assert len(traced_ops) >= 10


def test_mapped_symbol_coverage_accounting():
    """Track that expected mapped symbols appear in traced node targets."""
    x = torch.randn(4, 5)
    y = torch.randn(5, 3)
    x3 = torch.randn(2, 3, 4)
    y3 = torch.randn(2, 4, 5)
    idx = torch.tensor([[0, 1], [2, 3], [1, 0], [4, 2]], dtype=torch.long)
    heaviside_values = torch.tensor(0.0)

    class MappedModuleStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 5)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.flatten = nn.Flatten()

        def forward(self, t):
            t = self.linear(t)
            t = self.relu(t)
            t = self.sigmoid(t)
            t = self.tanh(t)
            return self.flatten(t)

    traces = [
        _trace_graph_module(lambda t: torch.relu(t)),
        _trace_graph_module(lambda t: torch.sigmoid(t)),
        _trace_graph_module(lambda t: torch.tanh(t)),
        _trace_graph_module(lambda t: torch.exp(t)),
        _trace_graph_module(lambda t: torch.log(torch.abs(t) + 1.0)),
        _trace_graph_module(lambda t: torch.sqrt(torch.abs(t) + 1.0)),
        _trace_graph_module(lambda t: torch.abs(t)),
        _trace_graph_module(lambda t: torch.sin(t)),
        _trace_graph_module(lambda t: torch.cos(t)),
        _trace_graph_module(lambda t: torch.tan(t * 0.1)),
        _trace_graph_module(lambda t: torch.clamp(t, min=-0.5, max=0.5)),
        _trace_graph_module(lambda t: torch.heaviside(t, values=heaviside_values)),
        _trace_graph_module(lambda t: torch.neg(t)),
        _trace_graph_module(lambda t: torch.reciprocal(torch.abs(t) + 1.0)),
        _trace_graph_module(lambda t: torch.pow(t, 2.0)),
        _trace_graph_module(lambda t: torch.sum(t, dim=1)),
        _trace_graph_module(lambda t: torch.mean(t, dim=1)),
        _trace_graph_module(lambda t: torch.max(t, dim=1)[0]),
        _trace_graph_module(lambda t: torch.min(t, dim=1)[0]),
        _trace_graph_module(lambda t: torch.reshape(t, (2, 10))),
        _trace_graph_module(lambda t: torch.flatten(t)),
        _trace_graph_module(lambda t: torch.unsqueeze(t, dim=0)),
        _trace_graph_module(lambda t: torch.squeeze(torch.unsqueeze(t, dim=0), dim=0)),
        _trace_graph_module(lambda t: torch.transpose(torch.reshape(t, (2, 2, 5)), 1, 2)),
        _trace_graph_module(lambda t: torch.permute(torch.reshape(t, (2, 2, 5)), (0, 2, 1))),
        _trace_graph_module(lambda t: torch.gather(t, dim=1, index=idx)),
        _trace_graph_module(lambda t: t[:, 1:4]),
        _trace_graph_module(lambda a, b: torch.matmul(a, b)),
        _trace_graph_module(lambda a, b: torch.mm(a, b)),
        _trace_graph_module(lambda a, b: torch.bmm(a, b)),
        _trace_graph_module(lambda t: torch.cat([t, t], dim=1)),
        _trace_graph_module(lambda t: torch.concat([t, t], dim=1)),
        _trace_graph_module(lambda t: torch.split(t, 2, dim=1)),
        _trace_graph_module(lambda t: t + 2.0),
        _trace_graph_module(lambda t: t - 2.0),
        _trace_graph_module(lambda t: t * 2.0),
        _trace_graph_module(lambda t: t / 2.0),
        _trace_graph_module(lambda t: t.relu()),
        _trace_graph_module(lambda t: t.sigmoid()),
        _trace_graph_module(lambda t: t.tanh()),
        _trace_graph_module(lambda t: t.reshape(2, 2, 5).transpose(1, 2)),
        _trace_graph_module(lambda t: t.reshape(2, 2, 5).permute(0, 2, 1)),
        _trace_graph_module(lambda t: t.sum(dim=1)),
        _trace_graph_module(lambda t: t.mean(dim=1)),
        _trace_graph_module(lambda t: t.max(dim=1)[0]),
        _trace_graph_module(lambda t: t.min(dim=1)[0]),
        _trace_graph_module(nn.Linear(5, 3)),
        _trace_graph_module(nn.ReLU()),
        _trace_graph_module(nn.Sigmoid()),
        _trace_graph_module(nn.Tanh()),
        _trace_graph_module(nn.Flatten()),
        _trace_graph_module(MappedModuleStack()),
        _trace_graph_module(lambda a, wt, bt: F.linear(a, wt, bt)),
    ]

    # Execute traces once to ensure graph materialization behavior is exercised.
    _ = traces[0](x)
    _ = traces[27](x, y)
    _ = traces[29](x3, y3)

    seen_functions: set[object] = set()
    seen_methods: set[str] = set()
    seen_modules: set[type[torch.nn.Module]] = set()

    for traced in traces:
        f_targets, m_targets, mod_targets = _collect_trace_targets(traced)
        seen_functions.update(f_targets)
        seen_methods.update(m_targets)
        seen_modules.update(mod_targets)

    # Function targets with potential aliases in FX target normalization.
    _assert_any_present({torch.matmul}, seen_functions)
    _assert_any_present({torch.mm}, seen_functions)
    _assert_any_present({torch.bmm}, seen_functions)
    _assert_any_present({F.linear}, seen_functions)

    _assert_any_present({torch.add, operator.add}, seen_functions)
    _assert_any_present({torch.sub, operator.sub}, seen_functions)
    _assert_any_present({torch.mul, operator.mul}, seen_functions)
    _assert_any_present({torch.div, operator.truediv}, seen_functions)
    _assert_any_present({torch.pow, operator.pow}, seen_functions)

    _assert_any_present({torch.neg, operator.neg}, seen_functions)
    _assert_any_present({torch.reciprocal}, seen_functions)
    _assert_any_present({torch.relu}, seen_functions)
    _assert_any_present({torch.sigmoid}, seen_functions)
    _assert_any_present({torch.tanh}, seen_functions)
    _assert_any_present({torch.exp}, seen_functions)
    _assert_any_present({torch.log}, seen_functions)
    _assert_any_present({torch.sqrt}, seen_functions)
    _assert_any_present({torch.abs}, seen_functions)
    _assert_any_present({torch.clamp}, seen_functions)
    _assert_any_present({torch.heaviside}, seen_functions)
    _assert_any_present({torch.sin}, seen_functions)
    _assert_any_present({torch.cos}, seen_functions)
    _assert_any_present({torch.tan}, seen_functions)
    _assert_any_present({torch.sum}, seen_functions)
    _assert_any_present({torch.mean}, seen_functions)
    _assert_any_present({torch.max}, seen_functions)
    _assert_any_present({torch.min}, seen_functions)

    _assert_any_present({torch.cat, torch.concat}, seen_functions)
    _assert_any_present({torch.split}, seen_functions)
    _assert_any_present({torch.gather}, seen_functions)

    # Structural ops can appear as call_function or call_method depending on FX lowering.
    _assert_function_or_method_present({torch.transpose}, {"transpose"}, seen_functions, seen_methods)
    _assert_function_or_method_present({torch.permute}, {"permute"}, seen_functions, seen_methods)
    _assert_function_or_method_present({torch.reshape}, {"reshape"}, seen_functions, seen_methods)
    _assert_function_or_method_present({torch.flatten}, {"flatten"}, seen_functions, seen_methods)
    _assert_function_or_method_present({torch.unsqueeze}, {"unsqueeze"}, seen_functions, seen_methods)
    _assert_function_or_method_present({torch.squeeze}, {"squeeze"}, seen_functions, seen_methods)

    expected_method_targets = {
        "relu",
        "sigmoid",
        "tanh",
        "transpose",
        "permute",
        "reshape",
        "flatten",
        "sum",
        "mean",
        "max",
        "min",
    }
    assert expected_method_targets.issubset(seen_methods)

    expected_module_types = {
        nn.Linear,
        nn.ReLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Flatten,
    }
    assert expected_module_types.issubset(seen_modules)
