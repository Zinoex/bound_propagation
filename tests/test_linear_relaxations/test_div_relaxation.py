"""
Tests for compute_div_relaxation soundness.

Covers:
  1. All 9 regime combinations: (a: pos/neg/zero-crossing) × (b: pos/neg/zero-crossing).
  2. 1D and 2D output shapes; 1D and 2D batch shapes.
  3. For any (a, b) sampled within the domain, z = a/b is contained in the relaxation.
"""

from __future__ import annotations

import pytest
import torch

from bound_propagation.propagation.linear_relaxations.div import compute_div_relaxation

# ---------------------------------------------------------------------------
# Soundness helper
# ---------------------------------------------------------------------------


def _check_soundness(
    lower_a: torch.Tensor,
    upper_a: torch.Tensor,
    lower_b: torch.Tensor,
    upper_b: torch.Tensor,
    num_samples: int = 40,
    tol: float = 1e-4,
) -> None:
    """
    For every element in the tensors:

    - If b crosses zero: assert coefficients are 0 and biases are ±inf.
    - Otherwise: grid-sample ``num_samples²`` (a, b) pairs in
      ``[lower_a, upper_a] × [lower_b, upper_b]`` and assert that
      ``z = a/b`` is enclosed by the linear relaxation.
    """
    relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
    coeff_a_lower, coeff_b_lower = relaxation.coeffs_lower
    coeff_a_upper, coeff_b_upper = relaxation.coeffs_upper
    bias_lower = relaxation.bias_lower
    bias_upper = relaxation.bias_upper

    crosses_zero_b = (lower_b <= 0) & (upper_b >= 0)

    # Zero-crossing denominator: bounds must be ±inf with zero coefficients.
    if crosses_zero_b.any():
        assert torch.all(torch.isneginf(bias_lower[crosses_zero_b])), (
            "bias_lower must be -inf for zero-crossing denominator elements"
        )
        assert torch.all(torch.isposinf(bias_upper[crosses_zero_b])), (
            "bias_upper must be +inf for zero-crossing denominator elements"
        )
        assert torch.all(coeff_a_lower[crosses_zero_b] == 0)
        assert torch.all(coeff_b_lower[crosses_zero_b] == 0)
        assert torch.all(coeff_a_upper[crosses_zero_b] == 0)
        assert torch.all(coeff_b_upper[crosses_zero_b] == 0)

    valid = ~crosses_zero_b
    if not valid.any():
        return

    # Boolean indexing flattens to 1D — one entry per valid element.

    # Input
    flat_la = lower_a[valid]
    flat_ua = upper_a[valid]
    flat_lb = lower_b[valid]
    flat_ub = upper_b[valid]

    # Relaxation
    flat_cal = coeff_a_lower[valid]
    flat_cbl = coeff_b_lower[valid]
    flat_cau = coeff_a_upper[valid]
    flat_cbu = coeff_b_upper[valid]
    flat_bl = bias_lower[valid]
    flat_bu = bias_upper[valid]

    # a_samples[i, k] = lower_a[k] + t[i] * (upper_a[k] - lower_a[k])
    t = torch.linspace(0, 1, num_samples)
    a_samples = flat_la[None, :] + t[:, None] * (flat_ua - flat_la)[None, :]
    b_samples = flat_lb[None, :] + t[:, None] * (flat_ub - flat_lb)[None, :]

    # z[i, j, k] = a_samples[i, k] / b_samples[j, k]
    # a_grid: (num_samples, 1, n_valid), b_grid: (1, num_samples, n_valid)
    a_grid = a_samples[:, None, :]
    b_grid = b_samples[None, :, :]
    z = a_grid / b_grid

    # Evaluate linear bounds at each (a, b) pair.
    cal = flat_cal[None, None, :]
    cbl = flat_cbl[None, None, :]
    cau = flat_cau[None, None, :]
    cbu = flat_cbu[None, None, :]
    bl = flat_bl[None, None, :]
    bu = flat_bu[None, None, :]

    lb_val = cal * a_grid + cbl * b_grid + bl
    ub_val = cau * a_grid + cbu * b_grid + bu

    lower_violations = z < lb_val - tol
    upper_violations = z > ub_val + tol

    if lower_violations.any():
        max_viol = (lb_val - z)[lower_violations].max().item()
        pytest.fail(f"Lower bound violated by {max_viol:.6f}")
    if upper_violations.any():
        max_viol = (z - ub_val)[upper_violations].max().item()
        pytest.fail(f"Upper bound violated by {max_viol:.6f}")


# ---------------------------------------------------------------------------
# All 9 regime combinations
# ---------------------------------------------------------------------------

_REGIME_PARAMS = pytest.mark.parametrize(
    ("la", "ua", "lb", "ub"),
    [
        pytest.param(1.0, 3.0, 1.0, 4.0, id="a_pos_b_pos"),
        pytest.param(1.0, 3.0, -4.0, -1.0, id="a_pos_b_neg"),
        pytest.param(1.0, 3.0, -1.0, 2.0, id="a_pos_b_cross"),
        pytest.param(-3.0, -1.0, 1.0, 4.0, id="a_neg_b_pos"),
        pytest.param(-3.0, -1.0, -4.0, -1.0, id="a_neg_b_neg"),
        pytest.param(-3.0, -1.0, -1.0, 2.0, id="a_neg_b_cross"),
        pytest.param(-2.0, 2.0, 1.0, 4.0, id="a_cross_b_pos"),
        pytest.param(-2.0, 2.0, -4.0, -1.0, id="a_cross_b_neg"),
        pytest.param(-2.0, 2.0, -1.0, 2.0, id="a_cross_b_cross"),
    ],
)


class TestDivRelaxationRegimes:
    """Soundness for all 9 (a_regime × b_regime) combinations, using scalar tensors."""

    @_REGIME_PARAMS
    def test_soundness(self, la: float, ua: float, lb: float, ub: float) -> None:
        _check_soundness(
            torch.tensor([la]),
            torch.tensor([ua]),
            torch.tensor([lb]),
            torch.tensor([ub]),
        )

    @_REGIME_PARAMS
    def test_output_shapes_match_input(self, la: float, ua: float, lb: float, ub: float) -> None:
        lower_a = torch.tensor([la])
        upper_a = torch.tensor([ua])
        lower_b = torch.tensor([lb])
        upper_b = torch.tensor([ub])
        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        ca_l, cb_l = relaxation.coeffs_lower
        ca_u, cb_u = relaxation.coeffs_upper
        for tensor in [ca_l, cb_l, ca_u, cb_u, relaxation.bias_lower, relaxation.bias_upper]:
            assert tensor.shape == lower_a.shape


# ---------------------------------------------------------------------------
# Near-zero denominator (not crossing, but close)
# ---------------------------------------------------------------------------


class TestDivRelaxationNearZero:
    """Soundness when the denominator interval is close to zero but does not cross it."""

    def test_near_zero_positive_b(self) -> None:
        _check_soundness(
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([0.01]),
            torch.tensor([0.1]),
        )

    def test_near_zero_negative_b(self) -> None:
        _check_soundness(
            torch.tensor([1.0]),
            torch.tensor([3.0]),
            torch.tensor([-0.1]),
            torch.tensor([-0.01]),
        )

    def test_near_zero_both_sides_negative_a(self) -> None:
        _check_soundness(
            torch.tensor([-3.0]),
            torch.tensor([-1.0]),
            torch.tensor([0.05]),
            torch.tensor([0.2]),
        )


# ---------------------------------------------------------------------------
# Zero-width intervals
# ---------------------------------------------------------------------------


class TestDivRelaxationZeroWidth:
    """Zero-width intervals: the relaxation must contain the single exact value z = a/b."""

    @pytest.mark.parametrize(
        ("a0", "b0"),
        [
            pytest.param(3.0, 2.0, id="pos_pos"),
            pytest.param(3.0, -2.0, id="pos_neg"),
            pytest.param(-3.0, 2.0, id="neg_pos"),
            pytest.param(-3.0, -2.0, id="neg_neg"),
        ],
    )
    def test_exact_point(self, a0: float, b0: float) -> None:
        lower_a = upper_a = torch.tensor([a0])
        lower_b = upper_b = torch.tensor([b0])
        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        ca_l, cb_l = relaxation.coeffs_lower
        ca_u, cb_u = relaxation.coeffs_upper

        a = torch.tensor([a0])
        b = torch.tensor([b0])
        z = a0 / b0

        lb_val = (ca_l * a + cb_l * b + relaxation.bias_lower).item()
        ub_val = (ca_u * a + cb_u * b + relaxation.bias_upper).item()

        assert lb_val <= z + 1e-4, f"Lower bound {lb_val} > z={z}"
        assert ub_val >= z - 1e-4, f"Upper bound {ub_val} < z={z}"


# ---------------------------------------------------------------------------
# Dimensional shapes
# ---------------------------------------------------------------------------


class TestDivRelaxationDimensions:
    """Output shapes are correct and soundness holds across various tensor layouts."""

    def test_1d_output_all_valid_mixed_sign(self) -> None:
        """Shape (4,): covers all four sign combinations of (a, b) without crossing."""
        lower_a = torch.tensor([1.0, 1.0, -3.0, -3.0])
        upper_a = torch.tensor([3.0, 3.0, -1.0, -1.0])
        lower_b = torch.tensor([1.0, -4.0, 1.0, -4.0])
        upper_b = torch.tensor([4.0, -1.0, 4.0, -1.0])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (4,)
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_1d_output_with_crossing_element(self) -> None:
        """Shape (3,): one zero-crossing b element, two valid elements."""
        lower_a = torch.tensor([1.0, -2.0, 1.0])
        upper_a = torch.tensor([3.0, 2.0, 3.0])
        lower_b = torch.tensor([1.0, -1.0, -3.0])
        upper_b = torch.tensor([4.0, 2.0, -1.0])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (3,)
        assert torch.isneginf(relaxation.bias_lower[1])
        assert torch.isposinf(relaxation.bias_upper[1])
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_2d_output_shape(self) -> None:
        """Shape (2, 3): 2D output with mixed regimes and one crossing element."""
        lower_a = torch.tensor([[1.0, 1.0, -3.0], [-2.0, 1.0, -3.0]])
        upper_a = torch.tensor([[3.0, 3.0, -1.0], [2.0, 3.0, -1.0]])
        lower_b = torch.tensor([[1.0, -4.0, 1.0], [1.0, -1.0, -4.0]])
        upper_b = torch.tensor([[4.0, -1.0, 4.0], [4.0, 2.0, -1.0]])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (2, 3)
        assert relaxation.bias_upper.shape == (2, 3)
        # Element [1, 1] has b crossing zero.
        assert torch.isneginf(relaxation.bias_lower[1, 1])
        assert torch.isposinf(relaxation.bias_upper[1, 1])
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_1d_batch_1d_output(self) -> None:
        """Shape (B=2, D=3): leading batch dimension, each batch uses a different b regime."""
        lower_a = torch.tensor([[1.0, -3.0, -2.0], [1.0, -3.0, 1.0]])
        upper_a = torch.tensor([[3.0, -1.0, 2.0], [3.0, -1.0, 3.0]])
        lower_b = torch.tensor([[1.0, 1.0, 1.0], [-4.0, -4.0, -4.0]])
        upper_b = torch.tensor([[4.0, 4.0, 4.0], [-1.0, -1.0, -1.0]])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (2, 3)
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_2d_batch_1d_output(self) -> None:
        """Shape (B1=2, B2=2, D=2): two leading batch dimensions, uniform a_pos/b_pos."""
        lower_a = torch.ones(2, 2, 2)
        upper_a = 3.0 * torch.ones(2, 2, 2)
        lower_b = torch.ones(2, 2, 2)
        upper_b = 4.0 * torch.ones(2, 2, 2)

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (2, 2, 2)
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_1d_batch_2d_output(self) -> None:
        """Shape (B=2, D1=2, D2=2): batch + 2D output, mixed sign regimes."""
        lower_a = torch.tensor([[[1.0, -3.0], [-2.0, 1.0]], [[1.0, -3.0], [1.0, -2.0]]])
        upper_a = torch.tensor([[[3.0, -1.0], [2.0, 3.0]], [[3.0, -1.0], [3.0, 2.0]]])
        lower_b = torch.tensor([[[1.0, 1.0], [1.0, -4.0]], [[-4.0, -4.0], [1.0, 1.0]]])
        upper_b = torch.tensor([[[4.0, 4.0], [4.0, -1.0]], [[-1.0, -1.0], [4.0, 4.0]]])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (2, 2, 2)
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_2d_batch_2d_output(self) -> None:
        """Shape (B1=2, B2=2, D1=2, D2=2): two batch dims + 2D output, a_neg/b_neg throughout."""
        lower_a = -3.0 * torch.ones(2, 2, 2, 2)
        upper_a = -1.0 * torch.ones(2, 2, 2, 2)
        lower_b = -4.0 * torch.ones(2, 2, 2, 2)
        upper_b = -1.0 * torch.ones(2, 2, 2, 2)

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)
        assert relaxation.bias_lower.shape == (2, 2, 2, 2)
        _check_soundness(lower_a, upper_a, lower_b, upper_b)

    def test_mixed_batch_valid_and_crossing(self) -> None:
        """Shape (2, 2): mix of valid and zero-crossing b elements, checks per-element output."""
        lower_a = torch.tensor([[1.0, -3.0], [-2.0, 1.0]])
        upper_a = torch.tensor([[3.0, -1.0], [2.0, 3.0]])
        lower_b = torch.tensor([[1.0, 1.0], [-1.0, -2.0]])
        upper_b = torch.tensor([[4.0, 4.0], [2.0, -1.0]])

        relaxation = compute_div_relaxation(lower_a, upper_a, lower_b, upper_b)

        # Element [1, 0]: b crosses zero → ±inf
        assert torch.isneginf(relaxation.bias_lower[1, 0])
        assert torch.isposinf(relaxation.bias_upper[1, 0])

        # Element [0, 0]: b positive → finite bounds
        assert torch.isfinite(relaxation.bias_lower[0, 0])
        assert torch.isfinite(relaxation.bias_upper[0, 0])

        _check_soundness(lower_a, upper_a, lower_b, upper_b)
