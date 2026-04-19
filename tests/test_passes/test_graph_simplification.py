"""Tests for :class:`SimplificationPass`.

Each test builds a small function, traces it with
:class:`BoundPropagationTracer` using the default IBP registry, runs
:class:`SimplificationPass`, and verifies both graph-level structure
(which targets survived) and numeric equivalence on random inputs.
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.fx as fx

from bound_propagation.passes import MetadataPass, SimplificationPass
from bound_propagation.propagation.ibp import create_default_ibp_registry
from bound_propagation.tracer import BoundPropagationTracer


def _trace(fn):
    tracer = BoundPropagationTracer(create_default_ibp_registry())
    return tracer.trace(fn)


def _targets(gm: fx.GraphModule) -> list[object]:
    return [node.target for node in gm.graph.nodes if node.op in ("call_function", "call_method")]


def _assert_numeric_equivalence(fn, gm: fx.GraphModule, *example_inputs) -> None:
    expected = fn(*example_inputs)
    got = gm(*example_inputs)
    assert torch.allclose(expected, got, rtol=1e-5, atol=1e-6)


def _simplify(gm: fx.GraphModule) -> fx.GraphModule:
    return SimplificationPass().run(gm)


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------


def test_add_zero_removed():
    def fn(x):
        return x + 0

    gm = _trace(fn)
    _simplify(gm)
    assert operator.add not in _targets(gm)
    assert torch.add not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_zero_add_removed():
    def fn(x):
        return 0 + x

    gm = _trace(fn)
    _simplify(gm)
    assert operator.add not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_sub_zero_removed():
    def fn(x):
        return x - 0

    gm = _trace(fn)
    _simplify(gm)
    assert operator.sub not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_mul_one_removed():
    def fn(x):
        return x * 1

    gm = _trace(fn)
    _simplify(gm)
    assert operator.mul not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_one_mul_removed():
    def fn(x):
        return 1 * x

    gm = _trace(fn)
    _simplify(gm)
    assert operator.mul not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_div_one_removed():
    def fn(x):
        return x / 1

    gm = _trace(fn)
    _simplify(gm)
    assert operator.truediv not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_div_by_const_becomes_mul():
    def fn(x):
        return x / 4.0

    gm = _trace(fn)
    _simplify(gm)
    targets = _targets(gm)
    assert operator.truediv not in targets
    assert operator.mul in targets
    _assert_numeric_equivalence(fn, gm, torch.randn(5))


def test_div_by_zero_untouched():
    def fn(x):
        return x / 0.5

    gm = _trace(fn)
    _simplify(gm)
    # 0.5 is a safe constant (nonzero), so we should still have rewritten.
    assert operator.truediv not in _targets(gm)


def test_double_neg_removed():
    def fn(x):
        return -(-x)  # noqa: B002

    gm = _trace(fn)
    _simplify(gm)
    assert operator.neg not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_algebraic_identities_compose():
    """Chain of identities should all dissolve in one pass."""

    def fn(x):
        return ((x + 0) * 1 - 0) / 1

    gm = _trace(fn)
    _simplify(gm)
    assert _targets(gm) == []
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


# ---------------------------------------------------------------------------
# Self-product and non-linear identities
# ---------------------------------------------------------------------------


def test_self_mul_becomes_pow():
    def fn(x):
        return x * x

    gm = _trace(fn)
    _simplify(gm)
    targets = _targets(gm)
    assert operator.mul not in targets
    assert torch.pow in targets
    _assert_numeric_equivalence(fn, gm, torch.randn(4))


def test_distinct_mul_untouched():
    def fn(x, y):
        return x * y

    gm = _trace(fn)
    _simplify(gm)
    assert operator.mul in _targets(gm)
    assert torch.pow not in _targets(gm)


def test_log_of_exp_collapses():
    def fn(x):
        return torch.log(torch.exp(x))

    gm = _trace(fn)
    _simplify(gm)
    targets = _targets(gm)
    assert torch.log not in targets
    assert torch.exp not in targets
    _assert_numeric_equivalence(fn, gm, torch.randn(4))


def test_log_of_non_exp_untouched():
    def fn(x):
        return torch.log(x)

    gm = _trace(fn)
    _simplify(gm)
    assert torch.log in _targets(gm)


# ---------------------------------------------------------------------------
# Structural no-ops
# ---------------------------------------------------------------------------


def test_noop_reshape_needs_metadata_to_fire():
    def fn(x):
        return x.reshape(x.shape)  # same-shape reshape

    # ``x.shape`` traces to a ``getattr`` node that the registry doesn't
    # support; fall back to a static shape target to keep this test focused.
    def fn_static(x):
        return x.reshape(3, 4)

    example = torch.randn(3, 4)
    gm = _trace(fn_static)
    MetadataPass(gm).run(example, abstract_mask=[True])
    _simplify(gm)
    targets = _targets(gm)
    assert "reshape" not in targets
    assert torch.reshape not in targets
    _assert_numeric_equivalence(fn_static, gm, example)


def test_noop_reshape_skipped_without_metadata():
    def fn(x):
        return x.reshape(3, 4)

    gm = _trace(fn)
    _simplify(gm)  # no MetadataPass
    # Still present because shape info is missing.
    assert "reshape" in _targets(gm)


def test_reshape_that_changes_shape_is_preserved():
    def fn(x):
        return x.reshape(12)

    example = torch.randn(3, 4)
    gm = _trace(fn)
    MetadataPass(gm).run(example, abstract_mask=[True])
    _simplify(gm)
    assert "reshape" in _targets(gm)
    _assert_numeric_equivalence(fn, gm, example)


def test_squeeze_unsqueeze_cancels():
    def fn(x):
        return x.unsqueeze(1).squeeze(1)

    gm = _trace(fn)
    _simplify(gm)
    targets = _targets(gm)
    assert "squeeze" not in targets
    assert "unsqueeze" not in targets
    _assert_numeric_equivalence(fn, gm, torch.randn(3, 5))


def test_squeeze_unsqueeze_different_dims_untouched():
    def fn(x):
        return x.unsqueeze(1).squeeze(0)

    gm = _trace(fn)
    _simplify(gm)
    targets = _targets(gm)
    assert "squeeze" in targets
    assert "unsqueeze" in targets


def test_double_transpose_cancels():
    def fn(x):
        return x.transpose(0, 1).transpose(0, 1)

    gm = _trace(fn)
    _simplify(gm)
    assert "transpose" not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3, 4))


def test_double_transpose_reversed_dims_still_cancels():
    def fn(x):
        return x.transpose(1, 0).transpose(0, 1)

    gm = _trace(fn)
    _simplify(gm)
    assert "transpose" not in _targets(gm)
    _assert_numeric_equivalence(fn, gm, torch.randn(3, 4))


def test_double_transpose_different_dims_untouched():
    def fn(x):
        return x.transpose(0, 1).transpose(1, 2)

    gm = _trace(fn)
    _simplify(gm)
    assert "transpose" in _targets(gm)


# ---------------------------------------------------------------------------
# Fixed-point behaviour
# ---------------------------------------------------------------------------


def test_nested_rewrites_converge():
    """``log(exp(x * 1 + 0)) → x`` should resolve in one run."""

    def fn(x):
        return torch.log(torch.exp(x * 1 + 0))

    gm = _trace(fn)
    _simplify(gm)
    assert _targets(gm) == []
    _assert_numeric_equivalence(fn, gm, torch.randn(3))


def test_max_iterations_validation():
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        SimplificationPass(max_iterations=0)


def test_empty_graph_is_noop():
    def fn(x):
        return x

    gm = _trace(fn)
    _simplify(gm)
    assert _targets(gm) == []
    _assert_numeric_equivalence(fn, gm, torch.randn(3))
