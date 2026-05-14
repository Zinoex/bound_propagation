from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.matmul import (
    ForwardLBPMatmul,
)
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_matmul_abstract_times_constant() -> None:
    """Test matmul: abstract @ constant."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Matmul: [x0, x1] @ [[1], [2]] = [x0 + 2*x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0], [2.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds, weight)

    # Expected: x0 + 2*x1
    # At (x0, x1) = (1, 3): 1 + 6 = 7
    # At (x0, x1) = (2, 4): 2 + 8 = 10
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([7.0]))
    assert torch.allclose(upper, torch.tensor([10.0]))


def test_matmul_constant_times_abstract() -> None:
    """Test matmul: constant @ abstract."""
    # Region: x0 ∈ [1, 2], x1 ∈ [3, 4]
    # Matmul: [[1, 2], [3, 1]] @ [x0, x1] = [x0 + 2*x1, 3*x0 + x1]
    region = HyperRectangle(lower=torch.tensor([1.0, 3.0]), upper=torch.tensor([2.0, 4.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0, 2.0], [3.0, 1.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, weight, bounds)

    lower, upper = result.concretize()
    # First output: x0 + 2*x1
    # At (x0, x1) = (1, 3): 1 + 6 = 7
    # At (x0, x1) = (2, 4): 2 + 8 = 10
    # Second output: 3*x0 + x1
    # At (x0, x1) = (1, 3): 3 + 3 = 6
    # At (x0, x1) = (2, 4): 6 + 4 = 10
    assert torch.allclose(lower, torch.tensor([7.0, 6.0]))
    assert torch.allclose(upper, torch.tensor([10.0, 10.0]))


def test_matmul_2d_constant() -> None:
    """Test matmul with 2D weight matrix."""
    # Region: x0 ∈ [0, 1], x1 ∈ [0, 1]
    # Matmul: [x0, x1] @ [[2, 0], [0, 3]] = [2*x0, 3*x1]
    region = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([1.0, 1.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds, weight)

    lower, upper = result.concretize()
    # Expected: [2*x0, 3*x1]
    # At (x0, x1) = (0, 0): [0, 0]
    # At (x0, x1) = (1, 1): [2, 3]
    assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([2.0, 3.0]))


def test_matmul_negative_weights() -> None:
    """Test matmul with negative weights."""
    # Region: x0 ∈ [1, 2]
    # Matmul: [[1], [-2]] @ [x0] = [x0, -2*x0]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    weight = torch.tensor([[1.0], [-2.0]])

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, weight, bounds)

    lower, upper = result.concretize()
    # Expected: [x0, -2*x0]
    # At x0=1: [1, -2]
    # At x0=2: [2, -4]
    assert torch.allclose(lower, torch.tensor([1.0, -4.0]))
    assert torch.allclose(upper, torch.tensor([2.0, -2.0]))


# ---------------------------------------------------------------------------
# Abstract @ abstract (McCormick relaxation)
# ---------------------------------------------------------------------------


def _disjoint_linear_bounds(
    region: HyperRectangle,
    slicer: slice,
    output_shape: tuple[int, ...],
) -> LinearBounds:
    """Build linear bounds selecting the input slice and reshaping to ``output_shape``.

    Allows packaging a single region as two independent ``LinearBounds`` (one
    for A, one for B) that share the same underlying input.
    """
    dim = region.lower.numel()
    picked = torch.eye(dim)[slicer]  # shape (out_numel, dim)
    out_numel = picked.shape[0]
    assert out_numel == int(torch.tensor(output_shape).prod().item())

    linear = picked.reshape(*output_shape, dim)
    bias = torch.zeros(output_shape)
    return LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=linear,
        bias_lower=bias,
        linear_upper=linear,
        bias_upper=bias,
    )


def _sample_matmul_bounds(fn, region: HyperRectangle, num_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``fn`` on uniform samples from ``region`` and return min/max."""
    rand = torch.rand(num_samples, *region.lower.shape)
    samples = region.lower + rand * (region.upper - region.lower)
    outputs = torch.stack([fn(s) for s in samples], dim=0)
    return outputs.amin(dim=0), outputs.amax(dim=0)


def _assert_sound_matmul(
    result_bounds: LinearBounds,
    fn,
    region: HyperRectangle,
    *,
    num_samples: int = 1000,
    atol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample-check that concretized bounds contain the true function range."""
    lower, upper = result_bounds.concretize()
    empirical_min, empirical_max = _sample_matmul_bounds(fn, region, num_samples)
    assert torch.all(lower <= empirical_min + atol), f"Lower unsound: lower={lower}, empirical_min={empirical_min}"
    assert torch.all(upper >= empirical_max - atol), f"Upper unsound: upper={upper}, empirical_max={empirical_max}"
    return lower, upper


def test_matmul_abstract_abstract_matrix_matrix_sound() -> None:
    """2x2 @ 2x2 with both operands abstract and strictly positive: sound."""
    region = HyperRectangle(
        lower=torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8]),
        upper=torch.tensor([2.0, 3.0, 4.0, 5.0, 1.5, 1.6, 1.7, 1.8]),
    )
    bounds_a = _disjoint_linear_bounds(region, slice(0, 4), (2, 2))
    bounds_b = _disjoint_linear_bounds(region, slice(4, 8), (2, 2))

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x[:4].reshape(2, 2) @ x[4:].reshape(2, 2)

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds_a, bounds_b)
    lower, upper = _assert_sound_matmul(result, fn, region)
    assert lower.shape == (2, 2)
    assert upper.shape == (2, 2)


def test_matmul_abstract_abstract_crossing_zero_sound() -> None:
    """Both operands cross zero; soundness must still hold."""
    region = HyperRectangle(
        lower=torch.tensor([-1.0, -2.0, -0.5, 0.2, -1.5, 0.3]),
        upper=torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.2]),
    )
    bounds_a = _disjoint_linear_bounds(region, slice(0, 3), (1, 3))  # (1, 3)
    bounds_b = _disjoint_linear_bounds(region, slice(3, 6), (3, 1))  # (3, 1)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x[:3].reshape(1, 3) @ x[3:].reshape(3, 1)

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds_a, bounds_b)
    _assert_sound_matmul(result, fn, region)


def test_matmul_abstract_abstract_degenerate_is_exact() -> None:
    """When both inputs are fixed (zero-width intervals), matmul is exact."""
    region = HyperRectangle(
        lower=torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8]),
        upper=torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8]),
    )
    bounds_a = _disjoint_linear_bounds(region, slice(0, 4), (2, 2))
    bounds_b = _disjoint_linear_bounds(region, slice(4, 8), (2, 2))

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds_a, bounds_b)
    lower, upper = result.concretize()

    expected = region.lower[:4].reshape(2, 2) @ region.lower[4:].reshape(2, 2)
    assert torch.allclose(lower, expected, atol=1e-5)
    assert torch.allclose(upper, expected, atol=1e-5)


def test_matmul_abstract_abstract_batched_sound() -> None:
    """Batched matmul ``(B, M, K) @ (B, K, N)`` is sound."""
    # 2 batch elements, each 2x2 @ 2x2; pack everything into one region.
    region = HyperRectangle(
        lower=torch.tensor([0.0] * 16),
        upper=torch.tensor([1.0] * 16),
    )
    bounds_a = _disjoint_linear_bounds(region, slice(0, 8), (2, 2, 2))
    bounds_b = _disjoint_linear_bounds(region, slice(8, 16), (2, 2, 2))

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x[:8].reshape(2, 2, 2) @ x[8:].reshape(2, 2, 2)

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds_a, bounds_b)
    lower, upper = _assert_sound_matmul(result, fn, region)
    assert lower.shape == (2, 2, 2)


def test_matmul_abstract_abstract_vector_dot_product() -> None:
    """1-D inputs reduce to a scalar dot product per PyTorch matmul semantics."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x[:2] @ x[2:]

    region = HyperRectangle(
        lower=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        upper=torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    bounds_a = _disjoint_linear_bounds(region, slice(0, 2), (2,))
    bounds_b = _disjoint_linear_bounds(region, slice(2, 4), (2,))

    strategy = ForwardLBPMatmul()
    result = propagate(strategy, bounds_a, bounds_b)
    lower, upper = _assert_sound_matmul(result, fn, region)
    assert lower.shape == ()


def test_matmul_abstract_abstract_alpha_crown_sound() -> None:
    """End-to-end alpha-CROWN optimization over a matmul abstract@abstract.

    The matmul knobs (``matmul_eta_lower``, ``matmul_eta_upper``) must
    integrate with the alpha-CROWN loop without breaking soundness or
    making the default width worse.
    """
    from bound_propagation.passes import MetadataPass
    from bound_propagation.propagation import AlphaOptimizationConfig, ForwardLBPPropagator
    from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
    from bound_propagation.tracer import BoundPropagationTracer

    def fn(x: torch.Tensor) -> torch.Tensor:
        a = x[:4].reshape(2, 2)
        b = x[4:].reshape(2, 2)
        return a @ b

    region = HyperRectangle(
        lower=torch.tensor([-0.5, 0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.4]),
        upper=torch.tensor([0.5, 1.0, 1.2, 0.8, 1.2, 0.3, 1.1, 0.4]),
    )

    registry = create_default_forward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn)
    MetadataPass(gm).run(region.lower)

    plain = ForwardLBPPropagator(gm, registry=registry).propagate([region])
    lo_plain, up_plain = plain.concretize()

    optimized = ForwardLBPPropagator(
        gm,
        registry=registry,
        alpha_config=AlphaOptimizationConfig(enabled=True, iterations=6, lr=0.1),
    ).propagate([region])
    lo_opt, up_opt = optimized.concretize()

    # Soundness: evaluate true function on random samples
    rand = torch.rand(300, *region.lower.shape)
    samples = region.lower + rand * (region.upper - region.lower)
    for sample in samples:
        y = fn(sample)
        assert torch.all(lo_opt <= y + 1e-4)
        assert torch.all(y <= up_opt + 1e-4)

    assert (up_opt - lo_opt).sum() <= (up_plain - lo_plain).sum() + 1e-4
