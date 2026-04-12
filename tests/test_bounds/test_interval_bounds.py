"""
Tests for bound representation types.

DEPRECATED: Many tests use the old API where IntervalBounds and LinearBounds required a region parameter.
The new API no longer requires this. New tests in test_linear_relaxation.py and test_method_propagators.py
cover the updated functionality.
"""

import pytest
import torch

from bound_propagation.bounds import AbstractBounds, IntervalBounds


class TestIntervalBounds:
    """Tests for IntervalBounds class."""

    def test_create_interval_bounds(self):
        """Test creating interval bounds."""
        lower = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor([2.0, 3.0, 4.0])

        bounds = IntervalBounds(lower, upper)

        assert torch.allclose(bounds.lower, lower)
        assert torch.allclose(bounds.upper, upper)
        assert bounds.shape == (3,)

    def test_invalid_bounds_upper_less_than_lower(self):
        """Test that invalid bounds raise errors."""
        # Lower > upper
        with pytest.raises(ValueError, match="Lower bound must be <= upper bound"):
            IntervalBounds(
                lower=torch.tensor([2.0, 3.0]),
                upper=torch.tensor([1.0, 2.0]),
            )

    def test_invalid_bounds_shape_mismatch(self):
        # Shape mismatch
        with pytest.raises(ValueError, match="same shape"):
            IntervalBounds(
                lower=torch.tensor([1.0, 2.0]),
                upper=torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_concretize(self):
        """Test concretization to intervals."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])

        bounds = IntervalBounds(lower, upper)
        concrete_lower, concrete_upper = bounds.concretize()

        assert torch.allclose(concrete_lower, lower)
        assert torch.allclose(concrete_upper, upper)

    def test_clone(self):
        """Test cloning bounds."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])

        bounds = IntervalBounds(lower, upper)
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

        bounds = IntervalBounds(lower, upper)

        # Move to same device (should work)
        bounds_cpu = bounds.to("cpu")
        assert bounds_cpu.device.type == "cpu"
        assert torch.allclose(bounds_cpu.lower, lower)

    def test_width_and_center(self):
        """Test interval width and center properties."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 6.0])

        bounds = IntervalBounds(lower, upper)

        # Width = upper - lower
        expected_width = torch.tensor([2.0, 4.0])
        assert torch.allclose(bounds.width, expected_width)

        # Center = (lower + upper) / 2
        expected_center = torch.tensor([2.0, 4.0])
        assert torch.allclose(bounds.center, expected_center)

    def test_unbounded_like_tensor(self):
        """Test creating unbounded interval."""
        tensor = torch.tensor([1.0, 2.0])
        bounds = IntervalBounds.unbounded_like(tensor)  # ty:ignore[invalid-argument-type]

        assert bounds.shape == tensor.shape
        assert torch.all(torch.isinf(bounds.lower) & (bounds.lower < 0))
        assert torch.all(torch.isinf(bounds.upper) & (bounds.upper > 0))

    def test_unbounded_like_interval_bounds(self):
        """Test creating unbounded interval matching another IntervalBounds."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])
        original_bounds = IntervalBounds(lower, upper)

        bounds = IntervalBounds.unbounded_like(original_bounds)

        assert bounds.shape == original_bounds.shape
        assert torch.all(torch.isinf(bounds.lower) & (bounds.lower < 0))
        assert torch.all(torch.isinf(bounds.upper) & (bounds.upper > 0))

    def test_multi_dimensional(self):
        """Test bounds with multiple dimensions."""
        shape = (2, 3, 4)
        lower = torch.randn(shape)
        upper = lower + torch.rand(shape)  # Ensure upper > lower

        bounds = IntervalBounds(lower, upper)

        assert bounds.shape == shape
        assert torch.allclose(bounds.lower, lower)
        assert torch.allclose(bounds.upper, upper)

    def test_abstract_bounds_interface(self):
        """Test that IntervalBounds implements AbstractBounds interface."""
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([3.0, 4.0])

        bounds = IntervalBounds(lower, upper)

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
