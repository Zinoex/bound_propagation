"""Backward LBP matmul tests.

Focuses on the ``abstract @ abstract`` case implemented via
:class:`MatmulBothAbstractRelaxation`. The constant-operand branches of
:class:`BackwardLBPMatmul` are already exercised by the full-workflow and
composition tests in this package.

McCormick bounds are not exact on bilinear terms, so the checks here are
sample-based soundness + an exact check on zero-width inputs.
"""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, region

# ---------------------------------------------------------------------------
# abstract @ abstract (McCormick)
# ---------------------------------------------------------------------------


class TestBackwardLBPMatmulBothAbstract:
    def test_matrix_matrix_positive(self):
        """Strictly positive operands: McCormick is sound."""

        def fn(x):
            a = x[:4].reshape(2, 2)
            b = x[4:].reshape(2, 2)
            return a @ b

        r = region(
            [1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8],
            [2.0, 3.0, 4.0, 5.0, 1.5, 1.6, 1.7, 1.8],
        )
        lower, upper = assert_sound(fn, r, num_samples=500)
        assert lower.shape == (2, 2)
        assert upper.shape == (2, 2)

    def test_matrix_matrix_crossing_zero(self):
        """Both operands cross zero: soundness must still hold."""

        def fn(x):
            a = x[:6].reshape(2, 3)
            b = x[6:].reshape(3, 2)
            return a @ b

        r = region(
            [-1.0, -0.5, 0.5, -0.8, 0.2, -0.3, -1.5, 0.3, -0.2, 0.1, -0.4, 0.5],
            [1.0, 1.0, 1.5, 0.8, 1.2, 0.4, 0.5, 1.3, 0.2, 1.1, 0.6, 1.5],
        )
        assert_sound(fn, r, num_samples=500)

    def test_vector_outer_shape(self):
        """(1, K) @ (K, 1) -> (1, 1) scalar matmul."""

        def fn(x):
            a = x[:3].reshape(1, 3)
            b = x[3:].reshape(3, 1)
            return a @ b

        r = region([-1.0, -2.0, -0.5, 0.2, -1.5, 0.3], [1.0, 0.5, 1.5, 1.0, 0.5, 1.2])
        lower, upper = assert_sound(fn, r, num_samples=500)
        assert lower.shape == (1, 1)

    def test_same_node_square(self):
        """y = x @ x for square x — accumulate_a_terms must handle correctly."""

        def fn(x):
            a = x.reshape(2, 2)
            return a @ a

        r = region([1.0, 0.5, -0.5, 0.2], [2.0, 1.0, 0.5, 1.0])
        assert_sound(fn, r, num_samples=500)

    def test_same_node_symmetric(self):
        """y = x @ x with x crossing zero: McCormick handles it."""

        def fn(x):
            a = x.reshape(2, 2)
            return a @ a

        r = region([-1.0, -0.5, -0.5, -0.3], [1.0, 1.0, 0.5, 0.7])
        assert_sound(fn, r, num_samples=500)

    def test_batched_matmul(self):
        """Batched 2x2 @ 2x2 matmul."""

        def fn(x):
            a = x[:8].reshape(2, 2, 2)
            b = x[8:].reshape(2, 2, 2)
            return a @ b

        r = region([0.0] * 16, [1.0] * 16)
        lower, upper = assert_sound(fn, r, num_samples=500)
        assert lower.shape == (2, 2, 2)

    def test_degenerate_zero_width_is_tight(self):
        """Zero-width inputs: bounds collapse to the exact product."""

        def fn(x):
            a = x[:4].reshape(2, 2)
            b = x[4:].reshape(2, 2)
            return a @ b

        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8])
        r = region(values.tolist(), values.tolist())
        bounds = assert_exact(
            fn,
            r,
            (values[:4].reshape(2, 2) @ values[4:].reshape(2, 2)),
            (values[:4].reshape(2, 2) @ values[4:].reshape(2, 2)),
        )
        assert bounds is not None


# ---------------------------------------------------------------------------
# alpha-CROWN integration for matmul knobs
# ---------------------------------------------------------------------------


class TestBackwardLBPMatmulAlpha:
    def test_alpha_crown_sound_and_tightens(self):
        """Enabling alpha-CROWN must keep bounds sound and not worsen width."""
        from bound_propagation.passes import MetadataPass
        from bound_propagation.propagation import (
            AlphaOptimizationConfig,
            BackwardLBPPropagator,
        )
        from bound_propagation.propagation.backward_lbp import (
            create_default_backward_lbp_registry,
        )
        from bound_propagation.tracer import BoundPropagationTracer

        def fn(x):
            a = x[:4].reshape(2, 2)
            b = x[4:].reshape(2, 2)
            return a @ b

        r = region(
            [-1.0, -0.5, 0.2, 0.3, -0.8, 0.1, -0.4, 0.5],
            [1.0, 0.8, 1.5, 1.2, 0.5, 1.1, 0.6, 1.5],
        )
        registry = create_default_backward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(r.lower)

        plain = BackwardLBPPropagator(gm, registry=registry).propagate([r])
        lo_plain, up_plain = plain.concretize()

        optimized = BackwardLBPPropagator(
            gm,
            registry=registry,
            alpha_config=AlphaOptimizationConfig(enabled=True, iterations=8, lr=0.1),
        ).propagate([r])
        lo_opt, up_opt = optimized.concretize()

        # Monte-Carlo soundness on the optimized bounds
        rand = torch.rand(400, *r.lower.shape)
        samples = r.lower + rand * (r.upper - r.lower)
        for sample in samples:
            y = fn(sample)
            assert torch.all(lo_opt <= y + 1e-4)
            assert torch.all(y <= up_opt + 1e-4)

        # alpha-CROWN should never make the width strictly worse
        plain_width = (up_plain - lo_plain).sum()
        opt_width = (up_opt - lo_opt).sum()
        assert opt_width <= plain_width + 1e-4
