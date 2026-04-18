"""
Tests for ElementwiseForwardRelaxation.forward and
PairedForwardRelaxation.forward.

Adapted from legacy tests that lived on LinearBounds directly, extended with
edge cases for multi-region bounds, shared input IDs, and mixed-sign coefficients.
"""

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.elementwise import ElementwiseForwardRelaxation
from bound_propagation.propagation.forward_lbp.pairwise import PairedForwardRelaxation
from bound_propagation.propagation.linear_relaxations.elementwise import ElementwiseParams
from bound_propagation.propagation.linear_relaxations.pairwise import PairedParams
from bound_propagation.regions import HyperRectangle


class TestElementwiseLinearRelaxationForwardCompose:
    """Tests for ElementwiseForwardRelaxation.forward."""

    def test_basic_positive_alpha(self):
        """Positive alpha: lower uses lower input, upper uses upper input."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([1.0]),
                beta_lower=torch.tensor([0.5]),
                alpha_upper=torch.tensor([2.0]),
                beta_upper=torch.tensor([1.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        # Lower: 1*(2*x + 1) + 0.5 = 2*x + 1.5
        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[2.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([1.5]))
        # Upper: 2*(3*x + 2) + 1 = 6*x + 5
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[6.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([5.0]))

    def test_negative_alpha_swaps_bounds(self):
        """Negative alpha: lower uses upper input bound, upper uses lower input bound.

        alpha_lower must be more negative (numerically smaller) than alpha_upper so that
        the composed bounds remain ordered lower <= upper.
        """
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # alpha_lower < alpha_upper (both negative): -2 < -1
        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([-2.0]),
                beta_lower=torch.tensor([0.5]),
                alpha_upper=torch.tensor([-1.0]),
                beta_upper=torch.tensor([1.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        # Lower: (-2) uses upper input (wu=3), bias uses bu=2
        #   linear: -2 * 3 = -6;  bias: -2*2 + 0.5 = -3.5
        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[-6.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([-3.5]))
        # Upper: (-1) uses lower input (wl=2), bias uses bl=1
        #   linear: -1 * 2 = -2;  bias: -1*1 + 1 = 0
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[-2.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    def test_mixed_sign_alpha_element_wise(self):
        """Mixed-sign alpha: each output element independently picks lower or upper input."""
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # Each output row selects one input dimension
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            bias_lower=torch.tensor([0.0, 0.0]),
            linear_upper=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            bias_upper=torch.tensor([0.0, 0.0]),
        )

        # Output 0: alpha=+1 (positive), output 1: alpha=-1 (negative)
        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([1.0, -1.0]),
                beta_lower=torch.tensor([0.0, 0.0]),
                alpha_upper=torch.tensor([1.0, -1.0]),
                beta_upper=torch.tensor([0.0, 0.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        # Output 0 (+alpha): lower uses lower, upper uses upper
        #   lower[0] = +1 * [1, 0] = [1, 0]
        #   upper[0] = +1 * [2, 0] = [2, 0]
        # Output 1 (-alpha): lower uses upper, upper uses lower
        #   lower[1] = -1 * [0, 2] = [0, -2]
        #   upper[1] = -1 * [0, 1] = [0, -1]
        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[1.0, 0.0], [0.0, -2.0]]))
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[2.0, 0.0], [0.0, -1.0]]))

    def test_zero_alpha_zeroes_out_linear_terms(self):
        """Zero alpha: linear contribution vanishes, output depends only on beta."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[5.0]]),
            bias_lower=torch.tensor([3.0]),
            linear_upper=torch.tensor([[7.0]]),
            bias_upper=torch.tensor([4.0]),
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([0.0]),
                beta_lower=torch.tensor([1.0]),
                alpha_upper=torch.tensor([0.0]),
                beta_upper=torch.tensor([2.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        # Output only depends on beta; linear coefficients are zero
        assert torch.allclose(result.bias_lower, torch.tensor([1.0]))
        assert torch.allclose(result.bias_upper, torch.tensor([2.0]))
        if result.linear_lower is not None:
            assert torch.allclose(result.linear_lower, torch.zeros_like(result.linear_lower))
        if result.linear_upper is not None:
            assert torch.allclose(result.linear_upper, torch.zeros_like(result.linear_upper))

    def test_identity_relaxation_passthrough(self):
        """Alpha=1, beta=0: output bounds equal input bounds exactly."""
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        wl = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        wu = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        bl = torch.tensor([0.5, 1.5])
        bu = torch.tensor([1.0, 2.0])

        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=wl,
            bias_lower=bl,
            linear_upper=wu,
            bias_upper=bu,
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.ones(2),
                beta_lower=torch.zeros(2),
                alpha_upper=torch.ones(2),
                beta_upper=torch.zeros(2),
            )
        )

        result = relaxation.forward(input_bounds)

        assert result.linear_lower is not None
        assert result.linear_upper is not None

        assert torch.allclose(result.linear_lower, wl)
        assert torch.allclose(result.bias_lower, bl)
        assert torch.allclose(result.linear_upper, wu)
        assert torch.allclose(result.bias_upper, bu)

    def test_constant_input_bounds(self):
        """Input with no linear terms (constant): output is also constant."""
        input_bounds = LinearBounds(
            bias_lower=torch.tensor([2.0]),
            bias_upper=torch.tensor([3.0]),
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([2.0]),
                beta_lower=torch.tensor([1.0]),
                alpha_upper=torch.tensor([3.0]),
                beta_upper=torch.tensor([0.5]),
            )
        )

        result = relaxation.forward(input_bounds)

        # No linear terms propagate
        assert result.linear_lower is None
        assert result.linear_upper is None
        # Lower: 2*2 + 1 = 5;  Upper: 3*3 + 0.5 = 9.5
        assert torch.allclose(result.bias_lower, torch.tensor([5.0]))
        assert torch.allclose(result.bias_upper, torch.tensor([9.5]))

    def test_constant_input_with_negative_alpha(self):
        """Negative alpha with constant input: bias picks the opposing input bias."""
        input_bounds = LinearBounds(
            bias_lower=torch.tensor([2.0]),
            bias_upper=torch.tensor([3.0]),
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([-1.0]),
                beta_lower=torch.tensor([0.0]),
                alpha_upper=torch.tensor([-1.0]),
                beta_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        # Lower: 0*bias_lower + (-1)*bias_upper + 0 = -3
        assert torch.allclose(result.bias_lower, torch.tensor([-3.0]))
        # Upper: 0*bias_upper + (-1)*bias_lower + 0 = -2
        assert torch.allclose(result.bias_upper, torch.tensor([-2.0]))

    def test_multi_region_input_bounds(self):
        """Multi-region input: each region's linear terms are composed independently."""
        region1 = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        region2 = HyperRectangle(torch.tensor([2.0]), torch.tensor([4.0]))
        input_bounds = LinearBounds(
            regions=[region1, region2],
            linear_lower=[torch.tensor([[1.0]]), torch.tensor([[-1.0]])],
            bias_lower=torch.tensor([0.0]),
            linear_upper=[torch.tensor([[2.0]]), torch.tensor([[1.0]])],
            bias_upper=torch.tensor([3.0]),
            input_ids=[10, 20],
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([2.0]),
                beta_lower=torch.tensor([1.0]),
                alpha_upper=torch.tensor([3.0]),
                beta_upper=torch.tensor([0.5]),
            )
        )

        result = relaxation.forward(input_bounds)

        # alpha_lower=2>0: lower uses lower input coefficients
        # region1 lower: 2 * 1  = 2;  region2 lower: 2 * (-1) = -2
        assert len(result.linear_lowers) == 2
        assert torch.allclose(result.linear_lowers[0], torch.tensor([[2.0]]))
        assert torch.allclose(result.linear_lowers[1], torch.tensor([[-2.0]]))

        # alpha_upper=3>0: upper uses upper input coefficients
        # region1 upper: 3 * 2 = 6;  region2 upper: 3 * 1 = 3
        assert len(result.linear_uppers) == 2
        assert torch.allclose(result.linear_uppers[0], torch.tensor([[6.0]]))
        assert torch.allclose(result.linear_uppers[1], torch.tensor([[3.0]]))

        # bias_lower = 2*0.0 + 0*3.0 + 1.0 = 1.0
        # bias_upper = 3*3.0 + 0*0.0 + 0.5 = 9.5
        assert torch.allclose(result.bias_lower, torch.tensor([1.0]))
        assert torch.allclose(result.bias_upper, torch.tensor([9.5]))
        assert result.input_ids == [10, 20]

    def test_multidimensional_output(self):
        """Multi-output composition with 2D input region."""
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # Shape: (2 outputs, 2 inputs)
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[1.0, 2.0], [1.5, 2.5]]),
            bias_lower=torch.tensor([0.5, 1.0]),
            linear_upper=torch.tensor([[2.0, 3.0], [2.5, 3.5]]),
            bias_upper=torch.tensor([1.0, 2.0]),
        )

        # Per-output slopes: alpha_lower <= alpha_upper for each output so the composed bounds are valid
        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([1.0, 1.0]),
                beta_lower=torch.tensor([0.0, 0.5]),
                alpha_upper=torch.tensor([2.0, 3.0]),
                beta_upper=torch.tensor([0.5, 1.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        assert result.linear_lower is not None
        assert result.linear_lower.shape == (2, 2)
        assert result.linear_upper is not None
        assert result.linear_upper.shape == (2, 2)

        # Lower: [1*[1,2], 1*[1.5,2.5]] = [[1,2],[1.5,2.5]];  bias: [1*0.5+0, 1*1.0+0.5] = [0.5, 1.5]
        assert torch.allclose(result.linear_lower, torch.tensor([[1.0, 2.0], [1.5, 2.5]]))
        assert torch.allclose(result.bias_lower, torch.tensor([0.5, 1.5]))

        # Upper: [2*[2,3], 3*[2.5,3.5]] = [[4,6],[7.5,10.5]];  bias: [2*1.0+0.5, 3*2.0+1.0] = [2.5, 7.0]
        assert torch.allclose(result.linear_upper, torch.tensor([[4.0, 6.0], [7.5, 10.5]]))
        assert torch.allclose(result.bias_upper, torch.tensor([2.5, 7.0]))

    def test_regions_and_input_ids_preserved(self):
        """Regions and input_ids from input_bounds propagate unchanged to output."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        input_bounds = LinearBounds(
            regions=region,
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.0]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
            input_ids=[99],
        )

        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([1.0]),
                beta_lower=torch.tensor([0.0]),
                alpha_upper=torch.tensor([1.0]),
                beta_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(input_bounds)

        assert result.input_ids == [99]
        assert result.regions[0] is region

    def test_1d_batch_positive_alpha(self):
        """1D batch (B=2): each batch element has its own alpha, output shapes include batch dim.

        Linear terms have shape (B, D, I); alpha/beta have shape (B, D).  The composition
        must treat each batch element independently.
        """
        # region.lower shape (B=2, I=1): each batch element is a separate [0, 1] interval
        region = HyperRectangle(torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [1.0]]))
        # linear shape: (B=2, D=1, I=1), bias shape: (B=2, D=1)
        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=torch.tensor([[[1.0]], [[3.0]]]),
            bias_lower=torch.tensor([[0.0], [1.0]]),
            linear_upper=torch.tensor([[[2.0]], [[4.0]]]),
            bias_upper=torch.tensor([[1.0], [2.0]]),
        )

        # Different alpha per batch element; alpha_lower < alpha_upper element-wise
        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.tensor([[1.0], [2.0]]),
                beta_lower=torch.tensor([[0.5], [0.0]]),
                alpha_upper=torch.tensor([[3.0], [4.0]]),
                beta_upper=torch.tensor([[1.0], [0.5]]),
            )
        )

        result = relaxation.forward(input_bounds)

        assert result.linear_lower is not None
        assert result.linear_lower.shape == (2, 1, 1)
        assert result.linear_upper is not None
        assert result.linear_upper.shape == (2, 1, 1)
        assert result.bias_lower.shape == (2, 1)
        assert result.bias_upper.shape == (2, 1)

        # Batch 0: alpha_lower=1, alpha_upper=3
        #   linear_lower = 1*1 = 1;  bias_lower = 1*0 + 0.5 = 0.5
        #   linear_upper = 3*2 = 6;  bias_upper = 3*1 + 1 = 4
        # Batch 1: alpha_lower=2, alpha_upper=4
        #   linear_lower = 2*3 = 6;  bias_lower = 2*1 + 0 = 2
        #   linear_upper = 4*4 = 16;  bias_upper = 4*2 + 0.5 = 8.5
        assert torch.allclose(result.linear_lower, torch.tensor([[[1.0]], [[6.0]]]))
        assert torch.allclose(result.bias_lower, torch.tensor([[0.5], [2.0]]))
        assert torch.allclose(result.linear_upper, torch.tensor([[[6.0]], [[16.0]]]))
        assert torch.allclose(result.bias_upper, torch.tensor([[4.0], [8.5]]))

    def test_2d_batch_positive_alpha(self):
        """2D batch (B1=2, B2=2): two leading batch dimensions are preserved in output shapes."""
        # region.lower shape (B1=2, B2=2, I=1)
        region = HyperRectangle(torch.zeros(2, 2, 1), torch.ones(2, 2, 1))
        # linear shape: (2, 2, 1, 1), bias shape: (2, 2, 1)
        wl = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])  # varying per batch element
        wu = wl + 1  # upper = lower + 1 for easy gap verification
        bl = torch.zeros(2, 2, 1)
        bu = torch.ones(2, 2, 1)

        input_bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=wl,
            bias_lower=bl,
            linear_upper=wu,
            bias_upper=bu,
        )

        # Uniform alpha across batch for simplicity; per-element alpha is tested in 1d case
        relaxation = ElementwiseForwardRelaxation(
            params=ElementwiseParams(
                alpha_lower=torch.ones(2, 2, 1),
                beta_lower=torch.zeros(2, 2, 1),
                alpha_upper=2 * torch.ones(2, 2, 1),
                beta_upper=torch.ones(2, 2, 1),
            )
        )

        result = relaxation.forward(input_bounds)

        assert result.linear_lower is not None
        assert result.linear_lower.shape == (2, 2, 1, 1)
        assert result.linear_upper is not None
        assert result.linear_upper.shape == (2, 2, 1, 1)
        assert result.bias_lower.shape == (2, 2, 1)
        assert result.bias_upper.shape == (2, 2, 1)

        # Lower: 1 * wl + 0 = wl
        assert torch.allclose(result.linear_lower, wl)
        assert torch.allclose(result.bias_lower, torch.zeros(2, 2, 1))
        # Upper: 2 * wu + 1 = 2*(wl+1) + 1
        assert torch.allclose(result.linear_upper, 2 * wu)
        assert torch.allclose(result.bias_upper, 2 * bu + torch.ones(2, 2, 1))


class TestPairedForwardRelaxationForwardCompose:
    """Tests for PairedForwardRelaxation.forward."""

    def test_basic_positive_coeffs_distinct_input_ids(self):
        """Both coefficients positive, distinct input IDs: contributions remain separate."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        # x1 = 2*x0 + 1 (lower), 3*x0 + 2 (upper)
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[1],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )
        # x2 = 1*x0 + 0 (lower), 2*x0 + 1 (upper)
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[2],
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.0]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # z >= 1*x1 + 1*x2  (lower),  z <= 2*x1 + 3*x2 + 0.5 (upper)
        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0]),
                alpha_upper_a=torch.tensor([2.0]),
                alpha_lower_b=torch.tensor([1.0]),
                alpha_upper_b=torch.tensor([3.0]),
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.5]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert len(result.linear_lowers) == 2
        assert len(result.linear_uppers) == 2

        idx1 = result.input_ids.index(1)
        idx2 = result.input_ids.index(2)

        # Lower: 1*(2*x+1) + 1*(1*x+0) = 3*x + 1
        # -> linear[id=1]: 1*2=2;  linear[id=2]: 1*1=1;  bias: 1*1 + 1*0 + 0 = 1
        assert torch.allclose(result.linear_lowers[idx1], torch.tensor([[2.0]]))
        assert torch.allclose(result.linear_lowers[idx2], torch.tensor([[1.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([1.0]))

        # Upper: 2*(3*x+2) + 3*(2*x+1) + 0.5 = 12*x + 7.5
        # -> linear[id=1]: 2*3=6;  linear[id=2]: 3*2=6;  bias: 2*2 + 3*1 + 0.5 = 7.5
        assert torch.allclose(result.linear_uppers[idx1], torch.tensor([[6.0]]))
        assert torch.allclose(result.linear_uppers[idx2], torch.tensor([[6.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([7.5]))

    def test_negative_coefficients_swap_bounds(self):
        """Negative coefficient for one input: lower uses that input's upper bound."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[1],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[2],
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.0]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0]),
                alpha_upper_a=torch.tensor([1.0]),
                alpha_lower_b=torch.tensor([-2.0]),
                alpha_upper_b=torch.tensor([-1.0]),
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        idx1 = result.input_ids.index(1)
        idx2 = result.input_ids.index(2)

        # Lower: (+1)*(2*x+1) + (-2)*(upper of x2: 2*x+1) = 2*x+1 - 4*x - 2 = -2*x - 1
        # -> linear[id=1]: 1*2=2;  linear[id=2]: -2*2=-4;  bias: 1*1 + (-2)*1 + 0 = -1
        assert torch.allclose(result.linear_lowers[idx1], torch.tensor([[2.0]]))
        assert torch.allclose(result.linear_lowers[idx2], torch.tensor([[-4.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([-1.0]))

        # Upper: (+1)*(3*x+2) + (-1)*(lower of x2: 1*x+0) = 3*x+2 - x = 2*x + 2
        # -> linear[id=1]: 1*3=3;  linear[id=2]: -1*1=-1;  bias: 1*2 + (-1)*0 + 0 = 2
        assert torch.allclose(result.linear_uppers[idx1], torch.tensor([[3.0]]))
        assert torch.allclose(result.linear_uppers[idx2], torch.tensor([[-1.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([2.0]))

    def test_shared_input_id_accumulates_linear_terms(self):
        """Same input_id in both inputs (z = f(x,x)): linear terms add up into one entry."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[42],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[42],
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[4.0]]),
            bias_upper=torch.tensor([3.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0]),
                alpha_upper_a=torch.tensor([1.0]),
                alpha_lower_b=torch.tensor([1.0]),
                alpha_upper_b=torch.tensor([1.0]),
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        # Shared id: single merged entry
        assert len(result.linear_lowers) == 1
        assert len(result.linear_uppers) == 1
        assert result.input_ids == [42]

        # Lower: 1*wl1 + 1*wl2 = 2+1 = 3;  bias: 1*1 + 1*0.5 + 0 = 1.5
        assert torch.allclose(result.linear_lowers[0], torch.tensor([[3.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([1.5]))

        # Upper: 1*wu1 + 1*wu2 = 3+4 = 7;  bias: 1*2 + 1*3 + 0 = 5
        assert torch.allclose(result.linear_uppers[0], torch.tensor([[7.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([5.0]))

    def test_shared_input_id_with_negative_coeff(self):
        """Shared input_id + negative coefficient: lower/upper swap before accumulation."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[99],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([0.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([1.0]),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[99],
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.0]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([0.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0]),
                alpha_upper_a=torch.tensor([1.0]),
                alpha_lower_b=torch.tensor([-1.0]),
                alpha_upper_b=torch.tensor([-1.0]),
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert len(result.linear_lowers) == 1
        assert result.input_ids == [99]

        # Lower: (+1)*wl1 + (-1)*wu2 = 2 + (-1)*2 = 0;  bias: (+1)*0 + (-1)*0 + 0 = 0
        assert torch.allclose(result.linear_lowers[0], torch.tensor([[0.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([0.0]))

        # Upper: (+1)*wu1 + (-1)*wl2 = 3 + (-1)*1 = 2;  bias: (+1)*1 + (-1)*0 + 0 = 1
        assert torch.allclose(result.linear_uppers[0], torch.tensor([[2.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([1.0]))

    def test_both_inputs_constant(self):
        """Both inputs have no linear terms: output is also constant."""
        bounds1 = LinearBounds(
            bias_lower=torch.tensor([2.0]),
            bias_upper=torch.tensor([3.0]),
        )
        bounds2 = LinearBounds(
            bias_lower=torch.tensor([1.0]),
            bias_upper=torch.tensor([4.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0]),
                alpha_upper_a=torch.tensor([1.0]),
                alpha_lower_b=torch.tensor([2.0]),
                alpha_upper_b=torch.tensor([3.0]),
                bias_lower=torch.tensor([0.5]),
                bias_upper=torch.tensor([1.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert result.linear_lower is None
        assert result.linear_upper is None
        # Lower: 1*2 + 2*1 + 0.5 = 4.5
        assert torch.allclose(result.bias_lower, torch.tensor([4.5]))
        # Upper: 1*3 + 3*4 + 1.0 = 16.0
        assert torch.allclose(result.bias_upper, torch.tensor([16.0]))

    def test_one_input_constant(self):
        """One input constant, one with linear terms: only one region in output."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[5],
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )
        bounds2 = LinearBounds(
            bias_lower=torch.tensor([1.0]),
            bias_upper=torch.tensor([2.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([2.0]),
                alpha_upper_a=torch.tensor([3.0]),
                alpha_lower_b=torch.tensor([1.0]),
                alpha_upper_b=torch.tensor([1.0]),
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert len(result.linear_lowers) == 1
        assert result.input_ids == [5]

        # Lower: linear[5]: 2*2=4;  bias: 2*1 + 1*1 + 0 = 3
        assert torch.allclose(result.linear_lowers[0], torch.tensor([[4.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([3.0]))

        # Upper: linear[5]: 3*3=9;  bias: 3*2 + 1*2 + 0 = 8
        assert torch.allclose(result.linear_uppers[0], torch.tensor([[9.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([8.0]))

    def test_multidimensional_inputs(self):
        """Multi-dimensional coefficients with distinct input IDs."""
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        # Diagonal-like linear terms for two outputs
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[1],
            linear_lower=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            bias_lower=torch.tensor([0.0, 0.0]),
            linear_upper=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            bias_upper=torch.tensor([0.0, 0.0]),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[2],
            linear_lower=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            bias_lower=torch.tensor([0.0, 0.0]),
            linear_upper=torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            bias_upper=torch.tensor([0.0, 0.0]),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.tensor([1.0, 1.0]),
                alpha_upper_a=torch.tensor([2.0, 2.0]),
                alpha_lower_b=torch.tensor([1.0, 1.0]),
                alpha_upper_b=torch.tensor([3.0, 3.0]),
                bias_lower=torch.tensor([0.0, 0.0]),
                bias_upper=torch.tensor([1.0, 1.0]),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert len(result.linear_lowers) == 2
        assert result.linear_lowers[0].shape == (2, 2)

        idx1 = result.input_ids.index(1)
        idx2 = result.input_ids.index(2)

        # Lower: coeff1=1, coeff2=1 (both positive, use lower input)
        # linear[id=1]: 1 * [[1,0],[0,1]] = [[1,0],[0,1]]
        # linear[id=2]: 1 * [[0,1],[1,0]] = [[0,1],[1,0]]
        assert torch.allclose(result.linear_lowers[idx1], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        assert torch.allclose(result.linear_lowers[idx2], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([0.0, 0.0]))

        # Upper: coeff1=2, coeff2=3 (both positive, use upper input)
        # linear[id=1]: 2 * [[2,0],[0,2]] = [[4,0],[0,4]]
        # linear[id=2]: 3 * [[0,2],[2,0]] = [[0,6],[6,0]]
        assert torch.allclose(result.linear_uppers[idx1], torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
        assert torch.allclose(result.linear_uppers[idx2], torch.tensor([[0.0, 6.0], [6.0, 0.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([1.0, 1.0]))

    def test_1d_batch_distinct_input_ids(self):
        """PairedForwardRelaxation with 1D batch (B=2): per-batch coefficients, two distinct IDs."""
        # region.lower shape (B=2, I=1)
        region = HyperRectangle(torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [1.0]]))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[1],
            linear_lower=torch.tensor([[[1.0]], [[2.0]]]),  # (2, 1, 1)
            bias_lower=torch.zeros(2, 1),
            linear_upper=torch.tensor([[[2.0]], [[3.0]]]),
            bias_upper=torch.ones(2, 1),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[2],
            linear_lower=torch.tensor([[[1.0]], [[1.0]]]),
            bias_lower=torch.zeros(2, 1),
            linear_upper=torch.tensor([[[2.0]], [[2.0]]]),
            bias_upper=0.5 * torch.ones(2, 1),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.ones(2, 1),
                alpha_upper_a=2 * torch.ones(2, 1),
                alpha_lower_b=torch.ones(2, 1),
                alpha_upper_b=2 * torch.ones(2, 1),
                bias_lower=torch.zeros(2, 1),
                bias_upper=torch.zeros(2, 1),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        assert result.bias_lower.shape == (2, 1)
        assert result.bias_upper.shape == (2, 1)

        idx1 = result.input_ids.index(1)
        idx2 = result.input_ids.index(2)

        # Lower: 1*wl1 + 1*wl2;  bias: 1*0 + 1*0 + 0 = 0
        assert torch.allclose(result.linear_lowers[idx1], torch.tensor([[[1.0]], [[2.0]]]))
        assert torch.allclose(result.linear_lowers[idx2], torch.tensor([[[1.0]], [[1.0]]]))
        assert torch.allclose(result.bias_lower, torch.zeros(2, 1))

        # Upper: 2*wu1 + 2*wu2;  bias: 2*1 + 2*0.5 + 0 = 3
        assert torch.allclose(result.linear_uppers[idx1], torch.tensor([[[4.0]], [[6.0]]]))
        assert torch.allclose(result.linear_uppers[idx2], torch.tensor([[[4.0]], [[4.0]]]))
        assert torch.allclose(result.bias_upper, 3 * torch.ones(2, 1))

    def test_2d_batch_shared_input_id(self):
        """PairedForwardRelaxation with 2D batch (B1=2, B2=2): shared input_id merges per-batch."""
        region = HyperRectangle(torch.zeros(2, 2, 1), torch.ones(2, 2, 1))
        # Both bounds share input_id=99 (representing z = f(x, x))
        bounds1 = LinearBounds(
            regions=[region],
            input_ids=[99],
            linear_lower=torch.ones(2, 2, 1, 1),
            bias_lower=torch.zeros(2, 2, 1),
            linear_upper=2 * torch.ones(2, 2, 1, 1),
            bias_upper=torch.ones(2, 2, 1),
        )
        bounds2 = LinearBounds(
            regions=[region],
            input_ids=[99],
            linear_lower=torch.ones(2, 2, 1, 1),
            bias_lower=torch.zeros(2, 2, 1),
            linear_upper=2 * torch.ones(2, 2, 1, 1),
            bias_upper=torch.ones(2, 2, 1),
        )

        relaxation = PairedForwardRelaxation(
            params=PairedParams(
                alpha_lower_a=torch.ones(2, 2, 1),
                alpha_upper_a=2 * torch.ones(2, 2, 1),
                alpha_lower_b=torch.ones(2, 2, 1),
                alpha_upper_b=torch.ones(2, 2, 1),
                bias_lower=torch.zeros(2, 2, 1),
                bias_upper=torch.zeros(2, 2, 1),
            )
        )

        result = relaxation.forward(bounds1, bounds2)

        # Shared id: single merged linear term
        assert len(result.linear_lowers) == 1
        assert result.input_ids == [99]
        assert result.linear_lowers[0].shape == (2, 2, 1, 1)

        # Lower: 1*wl1 + 1*wl2 = 1+1 = 2;  bias: 1*0 + 1*0 + 0 = 0
        assert torch.allclose(result.linear_lowers[0], 2 * torch.ones(2, 2, 1, 1))
        assert torch.allclose(result.bias_lower, torch.zeros(2, 2, 1))

        # Upper: 2*wu1 + 1*wu2 = 4+2 = 6;  bias: 2*1 + 1*1 + 0 = 3
        assert torch.allclose(result.linear_uppers[0], 6 * torch.ones(2, 2, 1, 1))
        assert torch.allclose(result.bias_upper, 3 * torch.ones(2, 2, 1))
