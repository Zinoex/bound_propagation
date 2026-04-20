"""Composition tests for backward LBP: multi-operation chains.

Tests that bounds remain sound through complex compositions of
linear, nonlinear, shape, and reduction operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conftest import assert_exact, assert_sound, propagate_bound, region

# ---------------------------------------------------------------------------
# Linear chains
# ---------------------------------------------------------------------------


class TestLinearChains:
    def test_matmul_add_chain_exact(self):
        """y = W2 (W1 x + b1) + b2: must be exact."""
        W1 = torch.tensor([[1.0, 0.5], [-1.0, 2.0], [0.5, -0.5]])
        b1 = torch.tensor([1.0, -1.0])
        W2 = torch.tensor([[1.0, 2.0]])
        b2 = torch.tensor([3.0])

        def chain_fn(x):
            h = x @ W1 + b1
            return h @ W2.T + b2

        r = region([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        bounds = propagate_bound(chain_fn, r)

        # Since this is purely linear, linear_lower == linear_upper
        assert torch.allclose(bounds.linear_lower, bounds.linear_upper, atol=1e-5)

        # Verify soundness via sampling
        lower, upper = bounds.concretize()
        from .conftest import check_soundness

        check_soundness(chain_fn, r, lower, upper)

    def test_sub_and_neg_exact(self):
        """y = -(x - c): tests Sub + Neg composition."""
        c = torch.tensor([1.0, 2.0])
        r = region([0.0, 0.0], [3.0, 4.0])
        assert_exact(
            lambda x: -(x - c),
            r,
            torch.tensor([-2.0, -2.0]),  # -(3-1), -(4-2)
            torch.tensor([1.0, 2.0]),  # -(0-1), -(0-2)
        )

    def test_constant_sub_abstract(self):
        """y = c - x: exact bounds."""
        c = torch.tensor([10.0, 20.0])
        r = region([1.0, 2.0], [3.0, 4.0])
        assert_exact(
            lambda x: c - x,
            r,
            torch.tensor([7.0, 16.0]),  # 10-3, 20-4
            torch.tensor([9.0, 18.0]),  # 10-1, 20-2
        )

    def test_add_both_abstract_exact(self):
        """y = x[:2] + x[2:]: both abstract, exact."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def add_fn(x):
            return x[:2] + x[2:]

        bounds = propagate_bound(add_fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([4.0, 6.0]))  # 1+3, 2+4
        assert torch.allclose(upper, torch.tensor([12.0, 14.0]))  # 5+7, 6+8

    def test_sub_both_abstract_exact(self):
        """y = x[:2] - x[2:]."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def sub_fn(x):
            return x[:2] - x[2:]

        bounds = propagate_bound(sub_fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([-6.0, -6.0]))  # 1-7, 2-8
        assert torch.allclose(upper, torch.tensor([2.0, 2.0]))  # 5-3, 6-4

    def test_matmul_left_constant_exact(self):
        """y = W @ x: left-constant matmul."""
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        def fn(x):
            return torch.matmul(W, x)

        r = region([0.0, 0.0], [1.0, 1.0])
        bounds = propagate_bound(fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([3.0, 7.0]))  # 1+2, 3+4


# ---------------------------------------------------------------------------
# Nonlinear single-layer chains
# ---------------------------------------------------------------------------


class TestNonlinearChains:
    def test_relu_then_sum_sound(self):
        """y = sum(relu(x)): nonlinear + reduction."""

        def fn(x):
            return torch.sum(torch.relu(x))

        assert_sound(fn, region([-2.0, -1.0, 0.5], [1.0, 2.0, 3.0]))

    def test_sigmoid_then_mul_constant_sound(self):
        """y = sigmoid(x) * 2."""

        def fn(x):
            return torch.sigmoid(x) * 2.0

        assert_sound(fn, region([-3.0, 0.0], [3.0, 2.0]))

    def test_exp_then_sum_sound(self):
        def fn(x):
            return torch.sum(torch.exp(x))

        assert_sound(fn, region([-1.0, 0.0], [1.0, 2.0]))

    def test_tanh_then_abs_sound(self):
        def fn(x):
            return torch.abs(torch.tanh(x))

        assert_sound(fn, region([-2.0, -1.0], [2.0, 1.0]))


# ---------------------------------------------------------------------------
# Multi-layer nonlinear chains
# ---------------------------------------------------------------------------


class TestMultiLayerChains:
    def test_linear_relu_linear_sound(self):
        """Classic two-layer network: W2 relu(W1 x + b1) + b2."""

        def fn(x):
            W1 = torch.tensor([[1.0, -1.0], [2.0, 0.5], [-1.0, 1.0]])
            b1 = torch.tensor([0.5, -0.5])
            h = torch.relu(x @ W1 + b1)
            W2 = torch.tensor([[1.0], [-1.0]])
            return h @ W2

        assert_sound(fn, region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))

    def test_relu_sigmoid_chain_sound(self):
        """relu then sigmoid."""

        def fn(x):
            return torch.sigmoid(torch.relu(x))

        assert_sound(fn, region([-2.0, -1.0], [2.0, 3.0]))

    def test_three_layer_sound(self):
        """Three-layer network with mixed activations."""

        def fn(x):
            W1 = torch.tensor([[1.0, -0.5], [0.5, 1.0], [-1.0, 0.5]])
            h1 = torch.relu(x @ W1)
            W2 = torch.tensor([[1.0, 0.5, -1.0], [0.5, -0.5, 1.0]])
            h2 = torch.sigmoid(h1 @ W2)
            W3 = torch.tensor([[1.0], [-1.0], [0.5]])
            return h2 @ W3

        assert_sound(fn, region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))

    def test_nn_linear_module_sound(self):
        """nn.Linear module (not just F.linear)."""
        model = nn.Linear(3, 2, bias=True)

        # Set known weights for reproducibility
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]))
            model.bias.copy_(torch.tensor([0.5, -0.5]))

        assert_sound(model, region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))

    def test_nn_linear_relu_sound(self):
        """nn.Linear -> ReLU composition."""

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = SimpleNet()
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]))
            model.linear.bias.copy_(torch.tensor([0.5, -0.5]))

        assert_sound(model, region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Diamond / fan-out graphs
# ---------------------------------------------------------------------------


class TestFanoutGraphs:
    def test_x_plus_x_exact(self):
        """y = x + x: fan-out with accumulation, must be 2x."""
        r = region([1.0, 2.0], [3.0, 4.0])
        assert_exact(
            lambda x: x + x,
            r,
            torch.tensor([2.0, 4.0]),
            torch.tensor([6.0, 8.0]),
        )

    def test_triple_fanout_exact(self):
        """y = x + x + x: 3x."""
        r = region([1.0], [2.0])
        assert_exact(
            lambda x: x + x + x,
            r,
            torch.tensor([3.0]),
            torch.tensor([6.0]),
        )

    def test_diamond_relu_sigmoid(self):
        """y = relu(x) + sigmoid(x): two paths from same input."""

        def fn(x):
            return torch.relu(x) + torch.sigmoid(x)

        assert_sound(fn, region([-2.0, 0.0], [2.0, 3.0]))

    def test_diamond_mul_add(self):
        """y = x * 2 + x * 3 = x * 5: exact."""
        r = region([1.0, -1.0], [3.0, 2.0])
        assert_exact(
            lambda x: x * 2.0 + x * 3.0,
            r,
            torch.tensor([5.0, -5.0]),
            torch.tensor([15.0, 10.0]),
        )

    def test_x_minus_x_exact(self):
        """y = x - x = 0: exact zero."""
        r = region([1.0, 2.0], [3.0, 4.0])
        assert_exact(
            lambda x: x - x,
            r,
            torch.zeros(2),
            torch.zeros(2),
        )


# ---------------------------------------------------------------------------
# Shape + nonlinear compositions
# ---------------------------------------------------------------------------


class TestShapeNonlinearComposition:
    def test_reshape_relu_reshape_sound(self):
        """reshape -> relu -> reshape."""

        def fn(x):
            h = x.reshape(2, 2)
            h = torch.relu(h)
            return h.reshape(4)

        assert_sound(fn, region([-1.0, -2.0, 1.0, 2.0], [1.0, 0.0, 3.0, 4.0]))

    def test_cat_relu_sound(self):
        """cat two halves then relu."""

        def fn(x):
            a = x[:2]
            b = x[2:]
            return torch.relu(torch.cat([b, a], dim=0))

        assert_sound(fn, region([-1.0, -2.0, 0.5, 1.0], [1.0, 2.0, 3.0, 4.0]))

    def test_getitem_sigmoid_sum_sound(self):
        """Slice -> sigmoid -> sum: complex chain."""

        def fn(x):
            h = torch.sigmoid(x[1:3])
            return torch.sum(h)

        assert_sound(fn, region([-3.0, -2.0, -1.0, 0.0], [0.0, 1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_width_region(self):
        """Degenerate region [a, a]: bounds must be f(a) exactly."""
        r = region([2.0, 3.0], [2.0, 3.0])
        assert_exact(lambda x: x, r, torch.tensor([2.0, 3.0]), torch.tensor([2.0, 3.0]))

    def test_zero_width_nonlinear(self):
        """Degenerate region through nonlinear: must match f(a)."""
        r = region([1.0], [1.0])
        lower, upper = assert_sound(lambda x: torch.relu(x), r)
        assert torch.allclose(lower, torch.tensor([1.0]), atol=1e-5)
        assert torch.allclose(upper, torch.tensor([1.0]), atol=1e-5)

    def test_scalar_output(self):
        """Scalar output from sum."""
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        lower, upper = assert_sound(lambda x: torch.sum(x), r)
        assert lower.numel() == 1

    def test_large_dimension(self):
        """Larger input dimension (stress test)."""
        lo = torch.zeros(50)
        hi = torch.ones(50)
        r = region(lo.tolist(), hi.tolist())
        assert_sound(lambda x: torch.relu(x), r, num_samples=500)

    def test_negative_scale_exact(self):
        """Negative constant scaling: bounds flip."""
        r = region([1.0, 2.0], [3.0, 4.0])
        assert_exact(
            lambda x: x * (-1.0),
            r,
            torch.tensor([-3.0, -4.0]),
            torch.tensor([-1.0, -2.0]),
        )

    def test_deeply_nested_linear(self):
        """Many linear ops chained: must remain exact."""
        r = region([1.0, 2.0], [3.0, 4.0])

        def fn(x):
            y = x + torch.tensor([1.0, 0.0])
            y = y * 2.0
            y = -y
            y = y - torch.tensor([3.0, 3.0])
            y = y / 2.0
            return y

        # Manual: y = (-(x + [1,0]) * 2 - [3,3]) / 2
        # = (-2x - [2,0] - [3,3]) / 2
        # = -x - [2.5, 1.5]
        assert_exact(
            fn,
            r,
            torch.tensor([-5.5, -5.5]),  # -3 - 2.5, -4 - 1.5
            torch.tensor([-3.5, -3.5]),  # -1 - 2.5, -2 - 1.5
        )
