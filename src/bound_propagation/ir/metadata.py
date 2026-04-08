"""
Metadata classes for tensor information tracking in computation graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class DeviceType(StrEnum):
    """Device type for tensor computation."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass(frozen=True)
class TensorMetadata:
    """
    Metadata describing a tensor's properties without storing the actual data.

    This is used for shape and type inference through computation graphs.

    Attributes:
        shape: Tuple representing tensor dimensions. Can include -1 for unknown dimensions.
               Follows PyTorch convention: (batch, *features) or (batch, channels, *spatial)
        dtype: String representation of data type (e.g., "float32", "float64", "int64")
        device: Device type where tensor resides
        requires_grad: Whether tensor requires gradient computation
    """

    shape: tuple[int, ...]
    dtype: str = "float32"
    device: DeviceType = DeviceType.CPU
    requires_grad: bool = False

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not isinstance(self.shape, tuple):
            # Convert to tuple if needed
            object.__setattr__(self, "shape", tuple(self.shape))

        # Validate shape contains integers (including -1 for unknown)
        for dim in self.shape:
            if not isinstance(dim, int):
                raise TypeError(f"Shape dimensions must be integers, got {type(dim)}")
            if dim < -1:
                raise ValueError(f"Shape dimensions must be >= -1, got {dim}")

    @property
    def ndim(self) -> int:
        """Number of dimensions in the tensor."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """
        Total number of elements in the tensor.
        Returns -1 if any dimension is unknown (-1).
        """
        if -1 in self.shape:
            return -1

        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def is_compatible_with(self, other: TensorMetadata) -> bool:
        """
        Check if this metadata is compatible with another for operations.

        Compatible means:
        - Same number of dimensions OR broadcasting is possible
        - dtypes are compatible (both floating point or both integer)
        - Device types match
        """
        if self.device != other.device:
            return False

        # Check dtype compatibility (simplified)
        self_is_float = "float" in self.dtype
        other_is_float = "float" in other.dtype

        # Check shape compatibility (broadcasting rules)
        # For now, simplified check - could be extended
        return self_is_float == other_is_float

    def broadcast_with(self, other: TensorMetadata) -> TensorMetadata:
        """
        Compute the resulting metadata after broadcasting with another tensor.

        Follows NumPy/PyTorch broadcasting semantics:
        - Dimensions are aligned from the right
        - Each dimension pair must be equal or one must be 1
        - Result has the larger dimension
        """
        # Align shapes from the right
        shape1 = self.shape
        shape2 = other.shape

        # Pad shorter shape with 1s on the left
        max_ndim = max(len(shape1), len(shape2))
        shape1_padded = (1,) * (max_ndim - len(shape1)) + shape1
        shape2_padded = (1,) * (max_ndim - len(shape2)) + shape2

        # Compute broadcast shape
        result_shape = []
        for d1, d2 in zip(shape1_padded, shape2_padded, strict=True):
            if d1 == -1 or d2 == -1:
                # Unknown dimension
                result_shape.append(-1)
            elif d1 == d2:
                result_shape.append(d1)
            elif d1 == 1:
                result_shape.append(d2)
            elif d2 == 1:
                result_shape.append(d1)
            else:
                raise ValueError(
                    f"Cannot broadcast shapes {self.shape} and {other.shape}: "
                    f"dimension mismatch at position with {d1} and {d2}"
                )

        # Determine result dtype (promote to higher precision)
        result_dtype = self._promote_dtype(self.dtype, other.dtype)

        return TensorMetadata(
            shape=tuple(result_shape),
            dtype=result_dtype,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    @staticmethod
    def _promote_dtype(dtype1: str, dtype2: str) -> str:
        """
        Promote dtypes to higher precision following PyTorch rules.
        Simplified version - could be extended.
        """
        # Priority order for floating point
        float_priority = ["float64", "float32", "float16"]
        int_priority = ["int64", "int32", "int16", "int8"]

        # If one is float and one is int, promote to float
        dtype1_is_float = "float" in dtype1
        dtype2_is_float = "float" in dtype2

        if dtype1_is_float != dtype2_is_float:
            # Mixed float/int - promote to float
            return dtype1 if dtype1_is_float else dtype2

        # Both float or both int - choose higher precision
        if dtype1_is_float:
            priority = float_priority
        else:
            priority = int_priority

        for dtype in priority:
            if dtype in (dtype1, dtype2):
                return dtype

        # Default fallback
        return dtype1

    def with_shape(self, new_shape: Sequence[int]) -> TensorMetadata:
        """Create a new metadata instance with a different shape."""
        return TensorMetadata(
            shape=tuple(new_shape),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def with_dtype(self, new_dtype: str) -> TensorMetadata:
        """Create a new metadata instance with a different dtype."""
        return TensorMetadata(
            shape=self.shape,
            dtype=new_dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"TensorMetadata(shape={self.shape}, dtype={self.dtype}, device={self.device.value}{grad_str})"
