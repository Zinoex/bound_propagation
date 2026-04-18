import pytest
import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.regions import HyperRectangle


class TestLinearBounds:
    """Tests for LinearBounds class."""

    def test_create_linear_bounds(self):
        """Test creating linear bounds."""
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        linear_lower = torch.tensor([[1.0, 2.0]])
        bias_lower = torch.tensor([0.5])
        linear_upper = torch.tensor([[2.0, 3.0]])
        bias_upper = torch.tensor([1.5])

        bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )

        assert torch.allclose(bounds.bias_lower, bias_lower)
        assert torch.allclose(bounds.bias_upper, bias_upper)
        assert bounds.region is region

    def test_concretize_with_hyperrectangle(self):
        """Test concretization of linear bounds with hyperrectangle."""
        # Input region: x in [0, 1], y in [0, 1]
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # Linear bounds: lower = 1*x + 2*y + 0.5, upper = 2*x + 3*y + 1.5
        linear_lower = torch.tensor([[1.0, 2.0]])
        bias_lower = torch.tensor([0.5])
        linear_upper = torch.tensor([[2.0, 3.0]])
        bias_upper = torch.tensor([1.5])

        bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )

        lower, upper = bounds.concretize()

        # Lower bound: minimize 1*x + 2*y + 0.5 over [0,1] x [0,1]
        # Minimum is at x=0, y=0: 0 + 0 + 0.5 = 0.5
        assert torch.allclose(lower, torch.tensor([0.5]))

        # Upper bound: maximize 2*x + 3*y + 1.5 over [0,1] x [0,1]
        # Maximum is at x=1, y=1: 2 + 3 + 1.5 = 6.5
        assert torch.allclose(upper, torch.tensor([6.5]))

    def test_concretize_with_multiple_regions(self):
        """Test concretization of linear bounds composed from multiple regions."""
        region_x = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        region_y = HyperRectangle(torch.tensor([2.0]), torch.tensor([4.0]))

        bounds = LinearBounds(
            regions=[region_x, region_y],
            linear_lower=[torch.tensor([[2.0]]), torch.tensor([[-1.0]])],
            bias_lower=torch.tensor([0.5]),
            linear_upper=[torch.tensor([[3.0]]), torch.tensor([[1.5]])],
            bias_upper=torch.tensor([1.0]),
            input_ids=[11, 22],
        )

        lower, upper = bounds.concretize()

        # Lower: minimize 2x - y + 0.5 over x in [0, 1], y in [2, 4] => 0 - 4 + 0.5
        assert torch.allclose(lower, torch.tensor([-3.5]))
        # Upper: maximize 3x + 1.5y + 1.0 over x in [0, 1], y in [2, 4] => 3 + 6 + 1
        assert torch.allclose(upper, torch.tensor([10.0]))
        assert bounds.input_ids == [11, 22]

    def test_concretize_with_unflattened_scalar_input_axis(self):
        """Test concretization with a scalar input region and no flattened input axis."""
        region = HyperRectangle(torch.tensor(2.0), torch.tensor(3.0))

        bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=[torch.tensor([2.0])],
            bias_lower=torch.tensor([0.5]),
            linear_upper=[torch.tensor([4.0])],
            bias_upper=torch.tensor([1.0]),
        )

        lower, upper = bounds.concretize()

        assert torch.allclose(lower, torch.tensor([4.5]))
        assert torch.allclose(upper, torch.tensor([13.0]))

    def test_concretize_with_unflattened_matrix_input_axes(self):
        """Test concretization with matrix-shaped input axes kept unflattened."""
        region = HyperRectangle(
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=[torch.ones(1, 2, 2)],
            bias_lower=torch.tensor([0.0]),
            linear_upper=[2 * torch.ones(1, 2, 2)],
            bias_upper=torch.tensor([1.0]),
        )

        lower, upper = bounds.concretize()

        assert torch.allclose(lower, torch.tensor([6.0]))
        assert torch.allclose(upper, torch.tensor([21.0]))

    def test_concretize_with_batched_region_shape(self):
        """Test concretization with region shape interpreted as (*batch_dims, *input_dims)."""
        # region shape: (batch=2, input=2)
        region = HyperRectangle(
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        # bias shape: (batch=2, output=1)
        # linear shape: (batch=2, output=1, input=2)
        bounds = LinearBounds(
            regions=[region],
            input_ids=[0],
            linear_lower=[torch.tensor([[[1.0, -2.0]], [[3.0, 4.0]]])],
            bias_lower=torch.tensor([[0.5], [1.0]]),
            linear_upper=[torch.tensor([[[2.0, 1.0]], [[4.0, 5.0]]])],
            bias_upper=torch.tensor([[1.5], [2.0]]),
        )

        lower, upper = bounds.concretize()

        assert torch.allclose(lower, torch.tensor([[-3.5], [19.0]]))
        assert torch.allclose(upper, torch.tensor([[5.5], [34.0]]))

    def test_reject_flattened_input_axes_with_batched_region_shape(self):
        """Flattened input axes are rejected for batched region shapes."""
        region = HyperRectangle(
            torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]]),
        )

        with pytest.raises(ValueError, match="input axes must match input shape"):
            LinearBounds(
                regions=[region],
                input_ids=[0],
                linear_lower=[torch.tensor([[[1.0, -1.0, 2.0, -2.0]], [[2.0, 0.0, -1.0, 3.0]]])],
                bias_lower=torch.tensor([[0.0], [1.0]]),
                linear_upper=[torch.tensor([[[2.0, 1.0, 3.0, 1.0]], [[3.0, 1.0, 0.0, 4.0]]])],
                bias_upper=torch.tensor([[1.0], [2.0]]),
            )
