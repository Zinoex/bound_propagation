"""
Tests for bound representation types.
"""
import pytest
import torch

from bound_propagation.bounds import AbstractBounds, IntervalBounds, LinearBounds
from bound_propagation.regions import HyperRectangle


class TestHyperRectangle:
    """Tests for HyperRectangle region class."""

    def test_create_hyperrectangle(self):
        """Test creating a hyperrectangle."""
        lower = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor([2.0, 3.0, 4.0])

        region = HyperRectangle(lower, upper)

        assert torch.allclose(region.lower, lower)
        assert torch.allclose(region.upper, upper)
        assert region.shape == (3,)

    def test_invalid_hyperrectangle_raises(self):
        """Test that invalid hyperrectangle raises errors."""
        # Lower > upper
        with pytest.raises(ValueError, match="Lower bound must be <= upper bound"):
            HyperRectangle(
                lower=torch.tensor([2.0, 3.0]),
                upper=torch.tensor([1.0, 2.0]),
            )

        # Shape mismatch
        with pytest.raises(ValueError, match="same shape"):
            HyperRectangle(
                lower=torch.tensor([1.0, 2.0]),
                upper=torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_hyperrectangle_properties(self):
        """Test hyperrectangle properties."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 6.0])

        region = HyperRectangle(lower, upper)

        # Width
        expected_width = torch.tensor([2.0, 4.0])
        assert torch.allclose(region.width, expected_width)

        # Center
        expected_center = torch.tensor([2.0, 4.0])
        assert torch.allclose(region.center, expected_center)

    def test_from_eps(self):
        """Test creating hyperrectangle from epsilon."""
        center = torch.tensor([1.0, 2.0, 3.0])
        epsilon = 0.5

        region = HyperRectangle.from_eps(center, epsilon)

        assert torch.allclose(region.lower, center - epsilon)
        assert torch.allclose(region.upper, center + epsilon)


class TestIntervalBounds:
    """Tests for IntervalBounds class."""

    def test_create_interval_bounds(self):
        """Test creating interval bounds."""
        lower = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor([2.0, 3.0, 4.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        assert torch.allclose(bounds.lower, lower)
        assert torch.allclose(bounds.upper, upper)
        assert bounds.shape == (3,)
        assert bounds.region is region

    def test_invalid_bounds_raises_error(self):
        """Test that invalid bounds raise errors."""
        lower_good = torch.tensor([1.0, 2.0])
        upper_good = torch.tensor([2.0, 3.0])
        region = HyperRectangle(lower_good, upper_good)

        # Lower > upper
        with pytest.raises(ValueError, match="Lower bound must be <= upper bound"):
            IntervalBounds(
                region,
                lower=torch.tensor([2.0, 3.0]),
                upper=torch.tensor([1.0, 2.0]),
            )

        # Shape mismatch
        with pytest.raises(ValueError, match="same shape"):
            IntervalBounds(
                region,
                lower=torch.tensor([1.0, 2.0]),
                upper=torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_concretize(self):
        """Test concretization to intervals."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)
        concrete_lower, concrete_upper = bounds.concretize()

        assert torch.allclose(concrete_lower, lower)
        assert torch.allclose(concrete_upper, upper)

    def test_clone(self):
        """Test cloning bounds."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)
        cloned = bounds.clone()

        assert torch.allclose(cloned.lower, bounds.lower)
        assert torch.allclose(cloned.upper, bounds.upper)

        # Verify it's a deep copy of bounds (not region, which is immutable)
        lower[0] = 999.0
        assert not torch.allclose(cloned.lower, lower)

    def test_to_device(self):
        """Test moving bounds to device."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        # Move to same device (should work)
        bounds_cpu = bounds.to("cpu")
        assert bounds_cpu.device.type == "cpu"
        assert torch.allclose(bounds_cpu.lower, lower)

    def test_width_and_center(self):
        """Test interval width and center properties."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 6.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        # Width = upper - lower
        expected_width = torch.tensor([2.0, 4.0])
        assert torch.allclose(bounds.width, expected_width)

        # Center = (lower + upper) / 2
        expected_center = torch.tensor([2.0, 4.0])
        assert torch.allclose(bounds.center, expected_center)

    def test_contains(self):
        """Test containment checking."""
        lower = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor([3.0, 4.0, 5.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        # Values inside bounds
        inside = torch.tensor([2.0, 3.0, 4.0])
        assert torch.all(bounds.contains(inside))

        # Values outside bounds
        outside = torch.tensor([0.0, 5.0, 6.0])
        assert not torch.all(bounds.contains(outside))

        # Mixed
        mixed = torch.tensor([2.0, 5.0, 4.0])
        result = bounds.contains(mixed)
        assert result[0] and not result[1] and result[2]

    def test_intersection(self):
        """Test interval intersection."""
        region = HyperRectangle(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))

        bounds1 = IntervalBounds(
            region,
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([5.0, 6.0, 7.0]),
        )
        bounds2 = IntervalBounds(
            region,
            lower=torch.tensor([3.0, 1.0, 4.0]),
            upper=torch.tensor([7.0, 4.0, 6.0]),
        )

        intersection = bounds1.intersection(bounds2)

        # Intersection lower = max(lower1, lower2)
        expected_lower = torch.tensor([3.0, 2.0, 4.0])
        # Intersection upper = min(upper1, upper2)
        expected_upper = torch.tensor([5.0, 4.0, 6.0])

        assert torch.allclose(intersection.lower, expected_lower)
        assert torch.allclose(intersection.upper, expected_upper)

    def test_union(self):
        """Test interval union (hull)."""
        region = HyperRectangle(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))

        bounds1 = IntervalBounds(
            region,
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([5.0, 6.0, 7.0]),
        )
        bounds2 = IntervalBounds(
            region,
            lower=torch.tensor([3.0, 1.0, 4.0]),
            upper=torch.tensor([7.0, 4.0, 6.0]),
        )

        union = bounds1.union(bounds2)

        # Union lower = min(lower1, lower2)
        expected_lower = torch.tensor([1.0, 1.0, 3.0])
        # Union upper = max(upper1, upper2)
        expected_upper = torch.tensor([7.0, 6.0, 7.0])

        assert torch.allclose(union.lower, expected_lower)
        assert torch.allclose(union.upper, expected_upper)

    def test_from_tensor(self):
        """Test creating bounds from tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Zero epsilon - point interval
        bounds = IntervalBounds.from_tensor(tensor, epsilon=0.0)
        assert torch.allclose(bounds.lower, tensor)
        assert torch.allclose(bounds.upper, tensor)

        # Positive epsilon - interval around point
        epsilon = 0.5
        bounds = IntervalBounds.from_tensor(tensor, epsilon=epsilon)
        assert torch.allclose(bounds.lower, tensor - epsilon)
        assert torch.allclose(bounds.upper, tensor + epsilon)

    def test_unbounded(self):
        """Test creating unbounded interval."""
        shape = (2, 3)
        bounds = IntervalBounds.unbounded(shape)

        assert bounds.shape == shape
        assert torch.all(torch.isinf(bounds.lower) & (bounds.lower < 0))
        assert torch.all(torch.isinf(bounds.upper) & (bounds.upper > 0))

    def test_multi_dimensional(self):
        """Test bounds with multiple dimensions."""
        shape = (2, 3, 4)
        lower = torch.randn(shape)
        upper = lower + torch.rand(shape)  # Ensure upper > lower
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        assert bounds.shape == shape
        assert torch.allclose(bounds.lower, lower)
        assert torch.allclose(bounds.upper, upper)

    def test_abstract_bounds_interface(self):
        """Test that IntervalBounds implements AbstractBounds interface."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        bounds = IntervalBounds(region, lower, upper)

        # Verify it's an AbstractBounds
        assert isinstance(bounds, AbstractBounds)

        # Verify all required properties/methods exist
        assert hasattr(bounds, "lower")
        assert hasattr(bounds, "upper")
        assert hasattr(bounds, "shape")
        assert hasattr(bounds, "device")
        assert hasattr(bounds, "region")
        assert hasattr(bounds, "to")
        assert hasattr(bounds, "concretize")
        assert hasattr(bounds, "clone")


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

    def test_from_interval_bounds(self):
        """Test creating linear bounds from interval bounds."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        region = HyperRectangle(lower, upper)

        interval_bounds = IntervalBounds(region, lower, upper)
        linear_bounds = LinearBounds.from_interval_bounds(interval_bounds)

        assert torch.allclose(linear_bounds.bias_lower, lower)
        assert torch.allclose(linear_bounds.bias_upper, upper)
        assert linear_bounds.linear_lower is None
        assert linear_bounds.linear_upper is None
