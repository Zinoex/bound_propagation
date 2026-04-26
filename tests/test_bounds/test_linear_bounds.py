import pytest
import torch

from bound_propagation.bounds import LinearBounds, LinearCoefficient
from bound_propagation.linear_operators import DenseOperator
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


class TestLinearCoefficient:
    """Per-input ``LinearCoefficient`` dataclass."""

    def _coeff(self, output_features: int = 1, input_features: int = 2) -> LinearCoefficient:
        region = HyperRectangle(torch.zeros(input_features), torch.ones(input_features))
        lower = DenseOperator(torch.randn(output_features, input_features), output_shape=(output_features,))
        upper = DenseOperator(lower.tensor + 1.0, output_shape=(output_features,))
        return LinearCoefficient(region=region, lower=lower, upper=upper)

    def test_attributes_accessible(self) -> None:
        c = self._coeff(2, 3)
        assert isinstance(c.region, HyperRectangle)
        assert c.lower.output_shape == torch.Size((2,))
        assert c.upper.input_shape == torch.Size((3,))

    def test_is_exact_when_lower_is_upper(self) -> None:
        region = HyperRectangle(torch.zeros(3), torch.ones(3))
        op = DenseOperator(torch.randn(2, 3), output_shape=(2,))
        c = LinearCoefficient(region=region, lower=op, upper=op)
        assert c.is_exact is True

    def test_is_exact_false_when_distinct(self) -> None:
        c = self._coeff()
        assert c.is_exact is False

    def test_input_shape_mismatch_raises(self) -> None:
        region = HyperRectangle(torch.zeros(3), torch.ones(3))
        lower = DenseOperator(torch.randn(2, 3), output_shape=(2,))
        upper = DenseOperator(torch.randn(2, 4), output_shape=(2,))
        with pytest.raises(ValueError, match="input shapes must match"):
            LinearCoefficient(region=region, lower=lower, upper=upper)

    def test_dataclass_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        c = self._coeff()
        with pytest.raises(FrozenInstanceError):
            c.region = HyperRectangle(torch.zeros(2), torch.ones(2))  # type: ignore[misc]


class TestLinearBoundsDictPath:
    """The ``coefficients=`` constructor path produces equivalent bounds to the parallel-list path."""

    def _build_via_parallel(self) -> LinearBounds:
        region_x = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        region_y = HyperRectangle(torch.tensor([2.0]), torch.tensor([4.0]))
        return LinearBounds(
            regions=[region_x, region_y],
            linear_lower=[torch.tensor([[2.0]]), torch.tensor([[-1.0]])],
            bias_lower=torch.tensor([0.5]),
            linear_upper=[torch.tensor([[3.0]]), torch.tensor([[1.5]])],
            bias_upper=torch.tensor([1.0]),
            input_ids=[11, 22],
        )

    def _build_via_dict(self) -> LinearBounds:
        region_x = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        region_y = HyperRectangle(torch.tensor([2.0]), torch.tensor([4.0]))
        coefficients = {
            11: LinearCoefficient(
                region=region_x,
                lower=DenseOperator(torch.tensor([[2.0]]), output_shape=(1,)),
                upper=DenseOperator(torch.tensor([[3.0]]), output_shape=(1,)),
            ),
            22: LinearCoefficient(
                region=region_y,
                lower=DenseOperator(torch.tensor([[-1.0]]), output_shape=(1,)),
                upper=DenseOperator(torch.tensor([[1.5]]), output_shape=(1,)),
            ),
        }
        return LinearBounds(
            bias_lower=torch.tensor([0.5]),
            bias_upper=torch.tensor([1.0]),
            coefficients=coefficients,
        )

    def test_equivalent_concretization(self) -> None:
        parallel = self._build_via_parallel().concretize()
        dict_ = self._build_via_dict().concretize()
        assert torch.allclose(parallel.lower, dict_.lower)
        assert torch.allclose(parallel.upper, dict_.upper)

    def test_coefficients_property_returns_dict_with_correct_keys(self) -> None:
        bounds = self._build_via_dict()
        coeffs = bounds.coefficients
        assert set(coeffs.keys()) == {11, 22}
        assert isinstance(coeffs[11], LinearCoefficient)

    def test_coefficients_property_is_defensive_copy(self) -> None:
        bounds = self._build_via_dict()
        coeffs = bounds.coefficients
        coeffs.pop(11)
        # Internal dict unchanged.
        assert 11 in bounds.coefficients

    def test_input_ids_iteration_order(self) -> None:
        bounds = self._build_via_dict()
        assert bounds.input_ids == [11, 22]

    def test_legacy_accessors_still_work(self) -> None:
        bounds = self._build_via_dict()
        assert len(bounds.linear_lowers_op) == 2
        assert len(bounds.linear_uppers_op) == 2
        assert len(bounds.regions) == 2
        assert bounds.input_ids == [11, 22]

    def test_cannot_pass_both_paths(self) -> None:
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        coeff = LinearCoefficient(
            region=region,
            lower=DenseOperator(torch.tensor([[1.0]]), output_shape=(1,)),
            upper=DenseOperator(torch.tensor([[1.0]]), output_shape=(1,)),
        )
        with pytest.raises(ValueError, match="Pass either"):
            LinearBounds(
                bias_lower=torch.tensor([0.0]),
                bias_upper=torch.tensor([0.0]),
                coefficients={0: coeff},
                regions=[region],
                linear_lower=torch.tensor([[1.0]]),
                linear_upper=torch.tensor([[1.0]]),
                input_ids=[0],
            )

    def test_clone_preserves_coefficients_dict(self) -> None:
        bounds = self._build_via_dict()
        cloned = bounds.clone()
        assert cloned.input_ids == bounds.input_ids
        assert torch.allclose(cloned.bias_lower, bounds.bias_lower)
        # Operators are cloned (different identities).
        assert cloned.coefficients[11].lower is not bounds.coefficients[11].lower

    def test_to_preserves_coefficients_dict(self) -> None:
        bounds = self._build_via_dict()
        moved = bounds.to("cpu")
        assert moved.input_ids == bounds.input_ids
        assert moved.coefficients.keys() == bounds.coefficients.keys()


class TestLinearBoundsIsExact:
    """``LinearBounds.is_exact`` is True iff every coefficient is exact and biases coincide."""

    def _shared_op(self) -> DenseOperator:
        return DenseOperator(torch.tensor([[1.0, 2.0]]), output_shape=(1,))

    def test_exact_when_op_and_bias_shared(self) -> None:
        op = self._shared_op()
        bias = torch.tensor([0.5])
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        bounds = LinearBounds(
            bias_lower=bias,
            bias_upper=bias,
            coefficients={0: LinearCoefficient(region=region, lower=op, upper=op)},
        )
        assert bounds.is_exact is True

    def test_not_exact_when_distinct_ops(self) -> None:
        op1 = self._shared_op()
        op2 = DenseOperator(op1.tensor.clone(), output_shape=(1,))
        bias = torch.tensor([0.5])
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        bounds = LinearBounds(
            bias_lower=bias,
            bias_upper=bias,
            coefficients={0: LinearCoefficient(region=region, lower=op1, upper=op2)},
        )
        assert bounds.is_exact is False

    def test_not_exact_when_distinct_biases(self) -> None:
        op = self._shared_op()
        bias = torch.tensor([0.5])
        bias_copy = bias.clone()
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        bounds = LinearBounds(
            bias_lower=bias,
            bias_upper=bias_copy,
            coefficients={0: LinearCoefficient(region=region, lower=op, upper=op)},
        )
        assert bounds.is_exact is False

    def test_exact_with_constant_bound_no_coefficients(self) -> None:
        # A constant bound (no linear terms) is exact when biases coincide.
        bias = torch.tensor([0.5])
        bounds = LinearBounds(bias_lower=bias, bias_upper=bias, coefficients={})
        assert bounds.is_exact is True

    def test_constant_bound_distinct_biases_not_exact(self) -> None:
        bias = torch.tensor([0.5])
        bias_copy = bias.clone()
        bounds = LinearBounds(bias_lower=bias, bias_upper=bias_copy, coefficients={})
        assert bounds.is_exact is False
