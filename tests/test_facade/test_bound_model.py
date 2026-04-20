"""Tests for the :class:`BoundModel` facade."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bound_propagation import (
    AlphaOptimizationConfig,
    BoundModel,
    HyperRectangle,
    RegistryExtension,
    TargetRegistry,
)
from bound_propagation.propagation.backward_lbp import (
    BackwardLBPStrategy,
    create_default_backward_lbp_registry,
)
from bound_propagation.propagation.forward_lbp import (
    ForwardLBPStrategy,
    create_default_forward_lbp_registry,
)
from bound_propagation.propagation.ibp import ForwardIBPStrategy, create_default_ibp_registry

ALL_METHODS = ["ibp", "forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"]
LBP_METHODS = ["forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"]
DUAL_REGISTRY_METHODS = ["forward_backward_lbp", "crown_ibp"]


def _sample_bounds_are_sound(fn, region, lower, upper, *, n=500, atol=1e-4):
    rand = torch.rand(n, *region.lower.shape)
    samples = region.lower + rand * (region.upper - region.lower)
    for sample in samples:
        out = fn(sample)
        assert torch.all(lower <= out + atol), (lower, out)
        assert torch.all(out <= upper + atol), (upper, out)


def _sample_multi_input_sound(fn, regions, lower, upper, *, n=500, atol=1e-4):
    for _ in range(n):
        args = [r.lower + torch.rand_like(r.lower) * (r.upper - r.lower) for r in regions]
        out = fn(*args)
        assert torch.all(lower <= out + atol), (lower, out)
        assert torch.all(out <= upper + atol), (upper, out)


def _concretize(bounds):
    """Get concrete (lower, upper) from either IntervalBounds or LinearBounds."""
    if hasattr(bounds, "concretize"):
        return bounds.concretize()
    return bounds.lower, bounds.upper


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


@pytest.fixture
def small_region():
    return HyperRectangle(
        lower=torch.full((3,), -0.5),
        upper=torch.full((3,), 0.5),
    )


# ---------------------------------------------------------------------------
# End-to-end smoke tests per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_basic_end_to_end(method, small_model, small_region):
    example = torch.zeros(3)
    bm = BoundModel(small_model, dummy_inputs=(example,), method=method)
    (bounds,) = bm.propagate(small_region)
    lower, upper = _concretize(bounds)
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert torch.all(lower <= upper + 1e-6)
    _sample_bounds_are_sound(small_model, small_region, lower, upper)


def test_multi_input_function():
    def fn(x, y):
        return x + y * 2

    bm = BoundModel(
        fn,
        dummy_inputs=(torch.zeros(2), torch.zeros(2)),
        method="ibp",
    )
    region_x = HyperRectangle(lower=torch.tensor([-1.0, 0.0]), upper=torch.tensor([1.0, 2.0]))
    region_y = HyperRectangle(lower=torch.tensor([0.0, -1.0]), upper=torch.tensor([1.0, 1.0]))
    (bounds,) = bm.propagate(region_x, region_y)
    lower, upper = _concretize(bounds)
    assert torch.allclose(lower, torch.tensor([-1.0, -2.0]))
    assert torch.allclose(upper, torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_composition_of_elementary_functions(method):
    """sigmoid(exp(x) + tanh(x)) — composition of nonlinear elementary ops."""

    def fn(x):
        return torch.sigmoid(torch.exp(x) + torch.tanh(x))

    region = HyperRectangle(lower=torch.tensor([-0.5, -0.3]), upper=torch.tensor([0.3, 0.4]))
    bm = BoundModel(fn, dummy_inputs=(torch.zeros(2),), method=method)
    (bounds,) = bm.propagate(region)
    lower, upper = _concretize(bounds)
    assert torch.all(lower <= upper + 1e-5)
    _sample_bounds_are_sound(fn, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dag_branch_rejoin(method):
    """y = relu(x); z = y * y + y — y is consumed twice, making the graph a DAG."""

    def fn(x):
        y = torch.relu(x)
        return y * y + y

    region = HyperRectangle(lower=torch.tensor([-1.0, -0.5, 0.2]), upper=torch.tensor([0.5, 1.0, 1.5]))
    bm = BoundModel(fn, dummy_inputs=(torch.zeros(3),), method=method)
    (bounds,) = bm.propagate(region)
    lower, upper = _concretize(bounds)
    assert torch.all(lower <= upper + 1e-5)
    _sample_bounds_are_sound(fn, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dag_two_branches_parallel(method):
    """Branch split then recombine: a = f(x), b = g(x), y = a + b."""

    def fn(x):
        a = torch.sigmoid(x)
        b = torch.tanh(x)
        return a + b

    region = HyperRectangle(lower=torch.tensor([-1.5, -0.5]), upper=torch.tensor([0.5, 1.5]))
    bm = BoundModel(fn, dummy_inputs=(torch.zeros(2),), method=method)
    (bounds,) = bm.propagate(region)
    lower, upper = _concretize(bounds)
    assert torch.all(lower <= upper + 1e-5)
    _sample_bounds_are_sound(fn, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_batch_1d(method, small_model):
    """Input region carries a leading batch dimension.

    Only IBP and forward-LBP support batched inputs; the backward tape hard-codes
    ``batch_ndim=0``, so ``backward_lbp`` / ``crown_ibp`` are excluded here.
    """
    batch = 4
    region = HyperRectangle(
        lower=torch.full((batch, 3), -0.5),
        upper=torch.full((batch, 3), 0.5),
    )
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(batch, 3),), method=method)
    (bounds,) = bm.propagate(region)
    lower, upper = _concretize(bounds)
    assert lower.shape == (batch, 2)
    assert upper.shape == (batch, 2)
    assert torch.all(lower <= upper + 1e-5)
    _sample_bounds_are_sound(small_model, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_batch_2d(method, small_model):
    """Two leading batch dims (e.g. (seq, batch, features))."""
    region = HyperRectangle(
        lower=torch.full((2, 3, 3), -0.4),
        upper=torch.full((2, 3, 3), 0.4),
    )
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(2, 3, 3),), method=method)
    (bounds,) = bm.propagate(region)
    lower, upper = _concretize(bounds)
    assert lower.shape == (2, 3, 2)
    assert upper.shape == (2, 3, 2)
    assert torch.all(lower <= upper + 1e-5)
    _sample_bounds_are_sound(small_model, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_input_and_weight_hyperrectangles(method):
    """Both operands of matmul are abstract: input x and uncertain weight W.

    Weight layout follows nn.Linear batched convention: ``W`` has shape
    ``(batch, output, input)`` and ``x`` has shape ``(batch, input)``. The
    output has shape ``(batch, output)``.
    """
    batch, in_dim, out_dim = 2, 3, 4

    def fn(x, w):
        return (w @ x.unsqueeze(-1)).squeeze(-1)

    x_region = HyperRectangle(
        lower=torch.full((batch, in_dim), -1.0),
        upper=torch.full((batch, in_dim), 1.0),
    )
    w_region = HyperRectangle(
        lower=torch.full((batch, out_dim, in_dim), -0.3),
        upper=torch.full((batch, out_dim, in_dim), 0.3),
    )
    bm = BoundModel(
        fn,
        dummy_inputs=(torch.zeros(batch, in_dim), torch.zeros(batch, out_dim, in_dim)),
        method=method,
    )
    (bounds,) = bm.propagate(x_region, w_region)
    lower, upper = _concretize(bounds)
    assert lower.shape == (batch, out_dim)
    assert upper.shape == (batch, out_dim)
    assert torch.all(lower <= upper + 1e-5)
    _sample_multi_input_sound(fn, (x_region, w_region), lower, upper)


# ---------------------------------------------------------------------------
# Registry overrides
# ---------------------------------------------------------------------------


def test_registry_override_single(small_model, small_region):
    registry = create_default_ibp_registry()
    bm = BoundModel(
        small_model,
        dummy_inputs=(torch.zeros(3),),
        method="ibp",
        registry=registry,
    )
    assert bm.registries["ibp"] is registry


def test_registry_override_dict_dual(small_model, small_region):
    fwd = create_default_forward_lbp_registry()
    bwd = create_default_backward_lbp_registry()
    bm = BoundModel(
        small_model,
        dummy_inputs=(torch.zeros(3),),
        method="forward_backward_lbp",
        registry={"forward_lbp": fwd, "backward_lbp": bwd},
    )
    assert bm.registries["forward_lbp"] is fwd
    assert bm.registries["backward_lbp"] is bwd


def test_single_registry_for_dual_method_raises(small_model):
    with pytest.raises(ValueError, match="pass a mapping"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="forward_backward_lbp",
            registry=create_default_forward_lbp_registry(),
        )


def test_dict_missing_required_key(small_model):
    with pytest.raises(ValueError, match="missing"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="crown_ibp",
            registry={"ibp": create_default_ibp_registry()},
        )


def test_dict_with_unexpected_key(small_model):
    with pytest.raises(ValueError, match="extra"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="ibp",
            registry={"ibp": create_default_ibp_registry(), "forward_lbp": create_default_forward_lbp_registry()},
        )


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


class _StubIBP(ForwardIBPStrategy):
    def propagate_forward(self, node, ctx):  # pragma: no cover - not invoked
        raise NotImplementedError


class _StubForwardLBP(ForwardLBPStrategy):
    def propagate_forward(self, node, ctx):  # pragma: no cover - not invoked
        raise NotImplementedError


class _StubBackwardLBP(BackwardLBPStrategy):
    def build_relaxation(self, node, tape, bounds):  # pragma: no cover - not invoked
        raise NotImplementedError


class _DummyTarget:
    """Dummy type used as an extension target (never appears in traced graphs)."""


def test_extension_registers_on_all_required_registries(small_model):
    ext = RegistryExtension(
        targets=[_DummyTarget],
        ibp=_StubIBP(),
        backward_lbp=_StubBackwardLBP(),
    )
    bm = BoundModel(
        small_model,
        dummy_inputs=(torch.zeros(3),),
        method="crown_ibp",
        extensions=[ext],
    )
    assert bm.registries["ibp"].supports_target(_DummyTarget)
    assert bm.registries["backward_lbp"].supports_target(_DummyTarget)


def test_extension_only_required_strategy_needed(small_model):
    """For single-registry methods, the extension only needs that one strategy."""
    ext = RegistryExtension(targets=[_DummyTarget], ibp=_StubIBP())
    bm = BoundModel(
        small_model,
        dummy_inputs=(torch.zeros(3),),
        method="ibp",
        extensions=[ext],
    )
    assert bm.registries["ibp"].supports_target(_DummyTarget)


def test_extension_missing_required_strategy_raises(small_model):
    # crown_ibp needs both ibp and backward_lbp; we provide only ibp.
    ext = RegistryExtension(targets=[_DummyTarget], ibp=_StubIBP())
    with pytest.raises(ValueError, match="missing the 'backward_lbp' strategy"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="crown_ibp",
            extensions=[ext],
        )


def test_extension_no_targets_raises(small_model):
    ext = RegistryExtension(targets=[], ibp=_StubIBP())
    with pytest.raises(ValueError, match="no targets"):
        BoundModel(
            small_model,
            method="ibp",
            dummy_inputs=(torch.zeros(3),),
            extensions=[ext],
        )


# ---------------------------------------------------------------------------
# Alpha-CROWN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", LBP_METHODS)
def test_alpha_config_accepted(method, small_model, small_region):
    alpha = AlphaOptimizationConfig(enabled=True, iterations=2, lr=0.1)
    bm = BoundModel(
        small_model,
        dummy_inputs=(torch.zeros(3),),
        method=method,
        alpha=alpha,
    )
    (bounds,) = bm.propagate(small_region)
    lower, upper = _concretize(bounds)
    assert torch.all(lower <= upper + 1e-5)


def test_alpha_rejected_for_ibp(small_model):
    alpha = AlphaOptimizationConfig(enabled=True)
    with pytest.raises(ValueError, match="IBP does not support"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="ibp",
            alpha=alpha,
        )


# ---------------------------------------------------------------------------
# Misc error paths
# ---------------------------------------------------------------------------


def test_unknown_method(small_model):
    with pytest.raises(ValueError, match="Unknown method"):
        BoundModel(
            small_model,
            dummy_inputs=(torch.zeros(3),),
            method="nope",  # ty:ignore[invalid-argument-type]
        )


def test_mismatched_example_inputs_count():
    def fn(x, y):
        return x + y

    with pytest.raises(ValueError, match="has 1 tensor"):
        BoundModel(fn, dummy_inputs=(torch.zeros(2),), method="ibp")


def test_mismatched_propagate_region_count(small_model, small_region):
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(3),), method="ibp")
    with pytest.raises(ValueError, match="Expected 1 input region"):
        bm.propagate(small_region, small_region)


def test_required_registry_keys_property(small_model):
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(3),), method="crown_ibp")
    assert bm.required_registry_keys == ("ibp", "backward_lbp")


def test_graph_module_reuse_across_propagate_calls(small_model, small_region):
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(3),), method="ibp")
    gm_1 = bm.graph_module
    bm.propagate(small_region)
    bm.propagate(small_region)
    assert bm.graph_module is gm_1


def test_registries_property_returns_configured_keys(small_model):
    bm = BoundModel(small_model, dummy_inputs=(torch.zeros(3),), method="forward_backward_lbp")
    assert set(bm.registries) == {"forward_lbp", "backward_lbp"}
    assert isinstance(bm.registries["forward_lbp"], TargetRegistry)
