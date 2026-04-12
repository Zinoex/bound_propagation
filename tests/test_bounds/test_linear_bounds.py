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

        bounds = LinearBounds(region, linear_lower, bias_lower, linear_upper, bias_upper)

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

        bounds = LinearBounds(region, linear_lower, bias_lower, linear_upper, bias_upper)

        lower, upper = bounds.concretize()

        # Lower bound: minimize 1*x + 2*y + 0.5 over [0,1] x [0,1]
        # Minimum is at x=0, y=0: 0 + 0 + 0.5 = 0.5
        assert torch.allclose(lower, torch.tensor([0.5]))

        # Upper bound: maximize 2*x + 3*y + 1.5 over [0,1] x [0,1]
        # Maximum is at x=1, y=1: 2 + 3 + 1.5 = 6.5
        assert torch.allclose(upper, torch.tensor([6.5]))

    def test_forward_compose_basic(self):
        """Test forward composition with linear bounds."""
        # Input region: x in [0, 1]
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: y = 2*x + 1 (lower), y = 3*x + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: z = 1*y + 0.5 (lower), z = 2*y + 1 (upper)
        other_bounds = LinearBounds(
            region,  # Region doesn't matter for other in forward compose
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # Result: g ∘ f
        # Lower: 1*(2*x + 1) + 0.5 = 2*x + 1.5
        # Upper: 2*(3*x + 2) + 1 = 6*x + 5
        result = self_bounds.forward_compose(other_bounds)

        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[2.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([1.5]))
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[6.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([5.0]))
        assert result.region is self_bounds.region

    def test_forward_compose_negative_weights(self):
        """Test forward composition with negative weights."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: y = 2*x + 1 (lower), y = 3*x + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: z = -1*y + 0.5 (lower), z = -2*y + 1 (upper)
        # Negative weights should swap lower/upper usage
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[-1.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[-2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # Result:
        # Lower: -1*y where y is at its max, so -1*(3*x + 2) + 0.5 = -3*x - 1.5
        # Upper: -2*y where y is at its min, so -2*(2*x + 1) + 1 = -4*x - 1
        result = self_bounds.forward_compose(other_bounds)

        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[-3.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([-1.5]))
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[-4.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([-1.0]))

    def test_forward_compose_constant_self(self):
        """Test forward composition when self has constant bounds (no linear terms)."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: constant bounds y = 1 (lower), y = 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=None,
            bias_lower=torch.tensor([1.0]),
            linear_upper=None,
            bias_upper=torch.tensor([2.0]),
        )

        # other: z = 3*y + 0.5 (lower), z = 4*y + 1 (upper)
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[3.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[4.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # Result: z is constant too since y is constant
        # Lower: 3*1 + 0.5 = 3.5
        # Upper: 4*2 + 1 = 9
        result = self_bounds.forward_compose(other_bounds)

        assert result.linear_lower is None
        assert torch.allclose(result.bias_lower, torch.tensor([3.5]))
        assert result.linear_upper is None
        assert torch.allclose(result.bias_upper, torch.tensor([9.0]))

    def test_forward_compose_constant_other(self):
        """Test forward composition when other has constant bounds."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: y = 2*x + 1 (lower), y = 3*x + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: z = 5 (lower), z = 7 (upper) - constant
        other_bounds = LinearBounds(
            region,
            linear_lower=None,
            bias_lower=torch.tensor([5.0]),
            linear_upper=None,
            bias_upper=torch.tensor([7.0]),
        )

        # Result: constant (other doesn't depend on y)
        result = self_bounds.forward_compose(other_bounds)

        assert result.linear_lower is None
        assert torch.allclose(result.bias_lower, torch.tensor([5.0]))
        assert result.linear_upper is None
        assert torch.allclose(result.bias_upper, torch.tensor([7.0]))

    def test_backward_compose_basic(self):
        """Test backward composition with linear bounds."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: z = 2*y + 1 (lower), z = 3*y + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: y = 1*x + 0.5 (lower), y = 2*x + 1 (upper)
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # Result: f ∘ g
        # Lower: 2*(1*x + 0.5) + 1 = 2*x + 2
        # Upper: 3*(2*x + 1) + 2 = 6*x + 5
        result = self_bounds.backward_compose(other_bounds)

        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[2.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([2.0]))
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[6.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([5.0]))
        assert result.region is other_bounds.region

    def test_backward_compose_negative_weights(self):
        """Test backward composition with negative weights."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: z = -2*y + 1 (lower), z = -3*y + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[-2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[-3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: y = 1*x + 0.5 (lower), y = 2*x + 1 (upper)
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[1.0]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[2.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        # Result:
        # Lower: -2*y where y is at its max, so -2*(2*x + 1) + 1 = -4*x - 1
        # Upper: -3*y where y is at its min, so -3*(1*x + 0.5) + 2 = -3*x + 0.5
        result = self_bounds.backward_compose(other_bounds)

        assert result.linear_lower is not None
        assert torch.allclose(result.linear_lower, torch.tensor([[-4.0]]))
        assert torch.allclose(result.bias_lower, torch.tensor([-1.0]))
        assert result.linear_upper is not None
        assert torch.allclose(result.linear_upper, torch.tensor([[-3.0]]))
        assert torch.allclose(result.bias_upper, torch.tensor([0.5]))

    def test_backward_compose_constant_self(self):
        """Test backward composition when self has constant bounds."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: z = 5 (lower), z = 7 (upper) - constant
        self_bounds = LinearBounds(
            region,
            linear_lower=None,
            bias_lower=torch.tensor([5.0]),
            linear_upper=None,
            bias_upper=torch.tensor([7.0]),
        )

        # other: y = 2*x + 1 (lower), y = 3*x + 2 (upper)
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # Result: constant (self doesn't depend on y)
        result = self_bounds.backward_compose(other_bounds)

        assert result.linear_lower is None
        assert torch.allclose(result.bias_lower, torch.tensor([5.0]))
        assert result.linear_upper is None
        assert torch.allclose(result.bias_upper, torch.tensor([7.0]))

    def test_backward_compose_constant_other(self):
        """Test backward composition when other has constant bounds."""
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # self: z = 2*y + 1 (lower), z = 3*y + 2 (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[2.0]]),
            bias_lower=torch.tensor([1.0]),
            linear_upper=torch.tensor([[3.0]]),
            bias_upper=torch.tensor([2.0]),
        )

        # other: y = 1 (lower), y = 2 (upper) - constant
        other_bounds = LinearBounds(
            region,
            linear_lower=None,
            bias_lower=torch.tensor([1.0]),
            linear_upper=None,
            bias_upper=torch.tensor([2.0]),
        )

        # Result: constant (y is constant)
        # Lower: 2*1 + 1 = 3
        # Upper: 3*2 + 2 = 8
        result = self_bounds.backward_compose(other_bounds)

        assert result.linear_lower is None
        assert torch.allclose(result.bias_lower, torch.tensor([3.0]))
        assert result.linear_upper is None
        assert torch.allclose(result.bias_upper, torch.tensor([8.0]))

    def test_forward_compose_multidimensional(self):
        """Test forward composition with multi-dimensional bounds."""
        # Input region: x in [0, 1]^2
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # self: y = [1, 2] @ x + [0.5, 1.0] (lower), y = [2, 3] @ x + [1.0, 2.0] (upper)
        self_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[1.0, 2.0], [1.5, 2.5]]),
            bias_lower=torch.tensor([0.5, 1.0]),
            linear_upper=torch.tensor([[2.0, 3.0], [2.5, 3.5]]),
            bias_upper=torch.tensor([1.0, 2.0]),
        )

        # other: z = [1, 0.5] @ y + 0.5 (lower), z = [2, 1] @ y + 1.0 (upper)
        other_bounds = LinearBounds(
            region,
            linear_lower=torch.tensor([[1.0, 0.5]]),
            bias_lower=torch.tensor([0.5]),
            linear_upper=torch.tensor([[2.0, 1.0]]),
            bias_upper=torch.tensor([1.0]),
        )

        result = self_bounds.forward_compose(other_bounds)

        # Verify result has correct shape
        assert result.linear_lower is not None
        assert result.linear_lower.shape == (1, 2)
        assert result.bias_lower.shape == (1,)
        assert result.linear_upper is not None
        assert result.linear_upper.shape == (1, 2)
        assert result.bias_upper.shape == (1,)

        # Lower: [1, 0.5] @ [[1, 2], [1.5, 2.5]] + [1, 0.5] @ [0.5, 1.0] + 0.5
        #      = [1*1 + 0.5*1.5, 1*2 + 0.5*2.5] + (1*0.5 + 0.5*1.0) + 0.5
        #      = [1.75, 3.25] + 1.0 + 0.5 = [1.75, 3.25] with bias 1.5
        expected_linear_lower = torch.tensor([[1.75, 3.25]])
        expected_bias_lower = torch.tensor([1.5])

        # Upper: [2, 1] @ [[2.0, 3.0], [2.5, 3.5]] + [2, 1] @ [1.0, 2.0] + 1.0
        #      = [2*2 + 1*2.5, 2*3 + 1*3.5] + (2*1.0 + 1*2.0) + 1.0
        #      = [6.5, 9.5] + 4.0 + 1.0 = [6.5, 9.5] with bias 5.0
        expected_linear_upper = torch.tensor([[6.5, 9.5]])
        expected_bias_upper = torch.tensor([5.0])

        assert torch.allclose(result.linear_lower, expected_linear_lower)
        assert torch.allclose(result.bias_lower, expected_bias_lower)
        assert torch.allclose(result.linear_upper, expected_linear_upper)
        assert torch.allclose(result.bias_upper, expected_bias_upper)
