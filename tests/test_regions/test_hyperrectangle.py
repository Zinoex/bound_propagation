import pytest
import torch

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

    def test_getitem_slices_hyperrectangle(self):
        """Test slicing a hyperrectangle."""
        lower = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor([4.0, 5.0, 6.0])

        region = HyperRectangle(lower, upper)

        sliced_region = region[1:]

        assert torch.allclose(sliced_region.lower, lower[1:])
        assert torch.allclose(sliced_region.upper, upper[1:])
        assert sliced_region.shape == (2,)
