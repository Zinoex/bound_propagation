"""
Tests for tensor metadata classes.
"""

from dataclasses import FrozenInstanceError

import pytest

from bound_propagation.ir import DeviceType, TensorMetadata


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types(self):
        """Test all device type values."""
        assert DeviceType.CPU == "cpu"
        assert DeviceType.CUDA == "cuda"
        assert DeviceType.MPS == "mps"

    def test_device_type_is_string(self):
        """Test that DeviceType values are strings."""
        assert isinstance(DeviceType.CPU, str)
        assert isinstance(DeviceType.CUDA, str)
        assert isinstance(DeviceType.MPS, str)


class TestTensorMetadata:
    """Tests for TensorMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        metadata = TensorMetadata(shape=(2, 3, 4))
        assert metadata.shape == (2, 3, 4)
        assert metadata.dtype == "float32"
        assert metadata.device == DeviceType.CPU
        assert metadata.requires_grad is False

    def test_with_all_parameters(self):
        """Test metadata creation with all parameters."""
        metadata = TensorMetadata(shape=(10, 20), dtype="float64", device=DeviceType.CUDA, requires_grad=True)
        assert metadata.shape == (10, 20)
        assert metadata.dtype == "float64"
        assert metadata.device == DeviceType.CUDA
        assert metadata.requires_grad is True

    def test_shape_conversion_to_tuple(self):
        """Test that shape is converted to tuple."""
        metadata = TensorMetadata(shape=[2, 3, 4])  # ty:ignore[invalid-argument-type]
        assert isinstance(metadata.shape, tuple)
        assert metadata.shape == (2, 3, 4)

    def test_ndim_property(self):
        """Test ndim property."""
        assert TensorMetadata(shape=(2,)).ndim == 1
        assert TensorMetadata(shape=(2, 3)).ndim == 2
        assert TensorMetadata(shape=(2, 3, 4)).ndim == 3
        assert TensorMetadata(shape=(2, 3, 4, 5, 6)).ndim == 5

    def test_numel_property(self):
        """Test numel property for known dimensions."""
        assert TensorMetadata(shape=(2,)).numel == 2
        assert TensorMetadata(shape=(2, 3)).numel == 6
        assert TensorMetadata(shape=(2, 3, 4)).numel == 24
        assert TensorMetadata(shape=(10, 20, 30)).numel == 6000

    def test_numel_with_unknown_dimension(self):
        """Test numel returns -1 when shape has unknown dimension."""
        assert TensorMetadata(shape=(-1, 3)).numel == -1
        assert TensorMetadata(shape=(2, -1, 4)).numel == -1
        assert TensorMetadata(shape=(-1,)).numel == -1

    def test_numel_with_zero_dimension(self):
        """Test numel with zero-sized dimensions."""
        assert TensorMetadata(shape=(0, 3)).numel == 0
        assert TensorMetadata(shape=(2, 0, 4)).numel == 0

    def test_shape_validation_non_integer(self):
        """Test that non-integer shape dimensions raise TypeError."""
        with pytest.raises(TypeError, match="Shape dimensions must be integers"):
            TensorMetadata(shape=(2.5, 3))  # ty:ignore[invalid-argument-type]

    def test_shape_validation_negative(self):
        """Test that invalid negative shape dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Shape dimensions must be >= -1"):
            TensorMetadata(shape=(2, -2))

    def test_shape_validation_allows_minus_one(self):
        """Test that -1 is allowed for unknown dimensions."""
        metadata = TensorMetadata(shape=(-1, 3, 4))
        assert metadata.shape == (-1, 3, 4)

    def test_frozen_dataclass(self):
        """Test that TensorMetadata is immutable."""
        metadata = TensorMetadata(shape=(2, 3))
        with pytest.raises(FrozenInstanceError):
            metadata.shape = (4, 5)  # ty:ignore[invalid-assignment]

    def test_is_compatible_with_same_device(self):
        """Test compatibility check with matching devices."""
        meta1 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta2 = TensorMetadata(shape=(2, 3), dtype="float64", device=DeviceType.CPU)
        assert meta1.is_compatible_with(meta2)

    def test_is_compatible_with_different_device(self):
        """Test compatibility check with different devices."""
        meta1 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta2 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CUDA)
        assert not meta1.is_compatible_with(meta2)

    def test_is_compatible_with_matching_dtypes(self):
        """Test compatibility check with matching dtype categories."""
        meta1 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta2 = TensorMetadata(shape=(2, 3), dtype="float64", device=DeviceType.CPU)
        assert meta1.is_compatible_with(meta2)

        meta3 = TensorMetadata(shape=(2, 3), dtype="int32", device=DeviceType.CPU)
        meta4 = TensorMetadata(shape=(2, 3), dtype="int64", device=DeviceType.CPU)
        assert meta3.is_compatible_with(meta4)

    def test_is_compatible_with_mismatched_dtype_categories(self):
        """Test compatibility check with different dtype categories."""
        meta_float = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta_int = TensorMetadata(shape=(2, 3), dtype="int32", device=DeviceType.CPU)
        assert not meta_float.is_compatible_with(meta_int)

    def test_broadcast_with_same_shape(self):
        """Test broadcasting with identical shapes."""
        meta1 = TensorMetadata(shape=(2, 3, 4))
        meta2 = TensorMetadata(shape=(2, 3, 4))
        result = meta1.broadcast_with(meta2)
        assert result.shape == (2, 3, 4)

    def test_broadcast_with_one_dimension(self):
        """Test broadcasting when one dimension is 1."""
        meta1 = TensorMetadata(shape=(2, 1, 4))
        meta2 = TensorMetadata(shape=(2, 3, 4))
        result = meta1.broadcast_with(meta2)
        assert result.shape == (2, 3, 4)

        # Symmetric case
        result2 = meta2.broadcast_with(meta1)
        assert result2.shape == (2, 3, 4)

    def test_broadcast_with_different_ndim(self):
        """Test broadcasting with different number of dimensions."""
        meta1 = TensorMetadata(shape=(3, 4))
        meta2 = TensorMetadata(shape=(2, 3, 4))
        result = meta1.broadcast_with(meta2)
        assert result.shape == (2, 3, 4)

        # Symmetric case
        result2 = meta2.broadcast_with(meta1)
        assert result2.shape == (2, 3, 4)

    def test_broadcast_with_scalar(self):
        """Test broadcasting with scalar (empty shape)."""
        meta_scalar = TensorMetadata(shape=())
        meta_tensor = TensorMetadata(shape=(2, 3, 4))
        result = meta_scalar.broadcast_with(meta_tensor)
        assert result.shape == (2, 3, 4)

    def test_broadcast_incompatible_shapes(self):
        """Test that incompatible shapes raise ValueError."""
        meta1 = TensorMetadata(shape=(2, 3))
        meta2 = TensorMetadata(shape=(2, 4))
        with pytest.raises(ValueError, match="Cannot broadcast"):
            meta1.broadcast_with(meta2)

    def test_broadcast_preserves_dtype(self):
        """Test that broadcasting preserves dtype and other properties."""
        meta1 = TensorMetadata(shape=(2, 1), dtype="float64", device=DeviceType.CUDA, requires_grad=True)
        meta2 = TensorMetadata(shape=(2, 3), dtype="float64", device=DeviceType.CUDA, requires_grad=False)
        result = meta1.broadcast_with(meta2)
        assert result.dtype == "float64"
        assert result.device == DeviceType.CUDA
        # requires_grad should be True if either is True
        assert result.requires_grad is True

    def test_batch_dimension_inference(self):
        """Test common batch dimension patterns."""
        # Batch of vectors
        batch_vec = TensorMetadata(shape=(32, 128))
        assert batch_vec.ndim == 2
        assert batch_vec.shape[0] == 32  # batch

        # Batch of images
        batch_img = TensorMetadata(shape=(32, 3, 224, 224))
        assert batch_img.ndim == 4
        assert batch_img.shape[0] == 32  # batch
        assert batch_img.shape[1] == 3  # channels

    def test_equality(self):
        """Test equality comparison."""
        meta1 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta2 = TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)
        meta3 = TensorMetadata(shape=(2, 3), dtype="float64", device=DeviceType.CPU)

        assert meta1 == meta2
        assert meta1 != meta3

    def test_repr(self):
        """Test string representation."""
        metadata = TensorMetadata(shape=(2, 3, 4), dtype="float32", device=DeviceType.CPU)
        repr_str = repr(metadata)
        assert "TensorMetadata" in repr_str
        assert "(2, 3, 4)" in repr_str
        assert "float32" in repr_str
