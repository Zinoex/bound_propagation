"""
Tests for LinearRelaxation data structure.
"""

import pytest
import torch

from bound_propagation.propagation.relaxations import LinearRelaxation


class TestLinearRelaxationConstruction:
    """Test construction and validation of LinearRelaxation."""

    def test_single_input_relaxation(self):
        """Test creating a simple single-input relaxation."""
        coeff_lower = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        coeff_upper = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        bias_lower = torch.tensor([0.0, 0.0])
        bias_upper = torch.tensor([0.0, 0.0])

        relaxation = LinearRelaxation(
            coeffs_lower=[coeff_lower],
            coeffs_upper=[coeff_upper],
            bias_lower=bias_lower,
            bias_upper=bias_upper,
        )

        assert relaxation.num_inputs == 1
        assert relaxation.bias_lower.shape == torch.Size([2])
        assert relaxation.bias_upper.shape == torch.Size([2])

    def test_multi_input_relaxation(self):
        """Test creating a multi-input relaxation (e.g., for ADD or MUL)."""
        # z = x + y
        coeff1_lower = torch.ones(3, 3)
        coeff2_lower = torch.ones(3, 3)
        bias_lower = torch.zeros(3)

        relaxation = LinearRelaxation(
            coeffs_lower=[coeff1_lower, coeff2_lower],
            coeffs_upper=[coeff1_lower, coeff2_lower],
            bias_lower=bias_lower,
            bias_upper=bias_lower,
        )

        assert relaxation.num_inputs == 2
        c1_l, c1_u = relaxation.get_input_coeff(0)
        assert torch.equal(c1_l, coeff1_lower)
        assert torch.equal(c1_u, coeff1_lower)

    def test_mismatched_coeffs_raises_error(self):
        """Test that mismatched coefficient counts raise an error."""
        with pytest.raises(ValueError, match="must match upper coefficients"):
            LinearRelaxation(
                coeffs_lower=[torch.ones(2, 2)],
                coeffs_upper=[torch.ones(2, 2), torch.ones(2, 2)],
                bias_lower=torch.zeros(2),
                bias_upper=torch.zeros(2),
            )

    def test_empty_coeffs_raises_error(self):
        """Test that empty coefficient list raises an error."""
        with pytest.raises(ValueError, match="At least one input"):
            LinearRelaxation(
                coeffs_lower=[],
                coeffs_upper=[],
                bias_lower=torch.zeros(2),
                bias_upper=torch.zeros(2),
            )

    def test_mismatched_bias_shapes_raises_error(self):
        """Test that mismatched bias shapes raise an error."""
        with pytest.raises(ValueError, match="Bias shapes don't match"):
            LinearRelaxation(
                coeffs_lower=[torch.ones(2, 2)],
                coeffs_upper=[torch.ones(2, 2)],
                bias_lower=torch.zeros(2),
                bias_upper=torch.zeros(3),
            )

    def test_with_metadata(self):
        """Test creating relaxation with shape metadata."""
        input_shape = torch.Size([2, 3])
        output_shape = torch.Size([2, 3])

        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(1)],
            coeffs_upper=[torch.ones(1)],
            bias_lower=torch.zeros(output_shape),
            bias_upper=torch.zeros(output_shape),
            input_shapes=[input_shape],
            output_shape=output_shape,
        )

        assert relaxation.input_shapes == [input_shape]
        assert relaxation.output_shape == output_shape

    def test_mismatched_input_shapes_raises_error(self):
        """Test that mismatched input shapes count raises an error."""
        with pytest.raises(ValueError, match="must match number of coefficients"):
            LinearRelaxation(
                coeffs_lower=[torch.ones(2, 2), torch.ones(2, 2)],
                coeffs_upper=[torch.ones(2, 2), torch.ones(2, 2)],
                bias_lower=torch.zeros(2),
                bias_upper=torch.zeros(2),
                input_shapes=[torch.Size([2, 2])],  # Only one shape, but two coeffs
            )


class TestLinearRelaxationMethods:
    """Test methods of LinearRelaxation."""

    def test_get_input_coeff(self):
        """Test retrieving coefficients for specific inputs."""
        coeff1_lower = torch.tensor([[1.0, 0.0]])
        coeff1_upper = torch.tensor([[2.0, 0.0]])
        coeff2_lower = torch.tensor([[0.0, 1.0]])
        coeff2_upper = torch.tensor([[0.0, 2.0]])

        relaxation = LinearRelaxation(
            coeffs_lower=[coeff1_lower, coeff2_lower],
            coeffs_upper=[coeff1_upper, coeff2_upper],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2),
        )

        c1_l, c1_u = relaxation.get_input_coeff(0)
        assert torch.equal(c1_l, coeff1_lower)
        assert torch.equal(c1_u, coeff1_upper)

        c2_l, c2_u = relaxation.get_input_coeff(1)
        assert torch.equal(c2_l, coeff2_lower)
        assert torch.equal(c2_u, coeff2_upper)

    def test_get_input_coeff_out_of_range(self):
        """Test that out-of-range index raises an error."""
        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(2, 2)],
            coeffs_upper=[torch.ones(2, 2)],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2),
        )

        with pytest.raises(IndexError, match="out of range"):
            relaxation.get_input_coeff(1)

        with pytest.raises(IndexError, match="out of range"):
            relaxation.get_input_coeff(-1)

    def test_to_device(self):
        """Test moving relaxation to different device."""
        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(2, 2)],
            coeffs_upper=[torch.ones(2, 2)],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2),
        )

        # Move to CPU (should work even if already on CPU)
        relaxation_cpu = relaxation.to(torch.device("cpu"))
        assert relaxation_cpu.coeffs_lower[0].device.type == "cpu"
        assert relaxation_cpu.bias_lower.device.type == "cpu"

    def test_is_exact_for_exact_relaxation(self):
        """Test is_exact returns True for exact relaxations."""
        coeff = torch.ones(2, 2)
        bias = torch.zeros(2)

        relaxation = LinearRelaxation(
            coeffs_lower=[coeff],
            coeffs_upper=[coeff],
            bias_lower=bias,
            bias_upper=bias,
        )

        assert relaxation.is_exact()

    def test_is_exact_for_inexact_relaxation(self):
        """Test is_exact returns False for approximate relaxations."""
        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(2, 2)],
            coeffs_upper=[torch.ones(2, 2) * 2],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2),
        )

        assert not relaxation.is_exact()

    def test_is_exact_with_tolerance(self):
        """Test is_exact with custom tolerance."""
        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(2, 2)],
            coeffs_upper=[torch.ones(2, 2) + 1e-9],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2) + 1e-9,
        )

        # Should be exact within default tolerance
        assert relaxation.is_exact()

        # Should not be exact with stricter tolerance
        assert not relaxation.is_exact(rtol=1e-12, atol=1e-12)


class TestLinearRelaxationFactoryMethods:
    """Test factory methods for creating common relaxations."""

    def test_create_identity(self):
        """Test creating an identity relaxation."""
        shape = torch.Size([2, 3])
        relaxation = LinearRelaxation.create_identity(shape, torch.device("cpu"), torch.float32)

        assert relaxation.num_inputs == 1
        assert relaxation.output_shape == shape
        assert relaxation.is_exact()

        # Identity should have coefficient 1 and bias 0
        coeff_l, coeff_u = relaxation.get_input_coeff(0)
        assert torch.equal(coeff_l, torch.ones(1))
        assert torch.equal(coeff_u, torch.ones(1))
        assert torch.all(relaxation.bias_lower == 0)
        assert torch.all(relaxation.bias_upper == 0)

    def test_create_diagonal_exact(self):
        """Test creating a diagonal relaxation with exact bounds."""
        alpha = torch.tensor([1.0, 2.0, 3.0])
        beta = torch.tensor([0.0, 0.0, 0.0])

        relaxation = LinearRelaxation.create_diagonal(
            alpha_lower=alpha,
            alpha_upper=alpha,
            beta_lower=beta,
            beta_upper=beta,
        )

        assert relaxation.num_inputs == 1
        assert relaxation.is_exact()
        assert torch.equal(relaxation.coeffs_lower[0], alpha)
        assert torch.equal(relaxation.bias_lower, beta)

    def test_create_diagonal_inexact(self):
        """Test creating a diagonal relaxation with approximate bounds."""
        alpha_lower = torch.tensor([0.5, 0.5, 0.5])
        alpha_upper = torch.tensor([1.0, 1.0, 1.0])
        beta_lower = torch.tensor([-0.1, -0.1, -0.1])
        beta_upper = torch.tensor([0.1, 0.1, 0.1])

        relaxation = LinearRelaxation.create_diagonal(
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            beta_lower=beta_lower,
            beta_upper=beta_upper,
        )

        assert relaxation.num_inputs == 1
        assert not relaxation.is_exact()
        assert torch.equal(relaxation.coeffs_lower[0], alpha_lower)
        assert torch.equal(relaxation.coeffs_upper[0], alpha_upper)
        assert torch.equal(relaxation.bias_lower, beta_lower)
        assert torch.equal(relaxation.bias_upper, beta_upper)


class TestLinearRelaxationProperties:
    """Test properties and invariants of LinearRelaxation."""

    def test_immutability(self):
        """Test that LinearRelaxation is immutable (frozen dataclass)."""
        relaxation = LinearRelaxation(
            coeffs_lower=[torch.ones(2, 2)],
            coeffs_upper=[torch.ones(2, 2)],
            bias_lower=torch.zeros(2),
            bias_upper=torch.zeros(2),
        )

        # Should not be able to assign to attributes
        with pytest.raises(AttributeError):
            relaxation.bias_lower = torch.ones(2)

    def test_num_inputs_property(self):
        """Test num_inputs property for various input counts."""
        for n in [1, 2, 5, 10]:
            relaxation = LinearRelaxation(
                coeffs_lower=[torch.ones(2, 2) for _ in range(n)],
                coeffs_upper=[torch.ones(2, 2) for _ in range(n)],
                bias_lower=torch.zeros(2),
                bias_upper=torch.zeros(2),
            )
            assert relaxation.num_inputs == n
