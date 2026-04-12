"""
Metadata classes for tensor information tracking in computation graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorMetadata:
    """
    Metadata describing a tensor's properties without storing the actual data.

    This is used for shape and type inference through computation graphs.

    Attributes:
        shape: Tuple representing tensor dimensions. Can include -1 for unknown dimensions.
               Follows PyTorch convention: (batch, *features) or (batch, channels, *spatial)
        dtype: String representation of data type (e.g., "torch.float32", "torch.float64", "torch.int64")
        device: Device where tensor resides
        requires_grad: Whether tensor requires gradient computation
    """

    shape: tuple[int, ...]
    dtype: str | torch.dtype = "float32"
    device: torch.Device = torch.device("cpu")
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

        object.__setattr__(self, "dtype", self._normalize_dtype(self.dtype))

        if self.device is None:
            raise TypeError("device must be a str, int, or torch.device, got None")

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
                raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}: dimension mismatch at position with {d1} and {d2}")

        # Determine result dtype (promote to higher precision)
        result_dtype = self._promote_dtype(self.dtype, other.dtype)

        return TensorMetadata(
            shape=tuple(result_shape),
            dtype=result_dtype,
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    @staticmethod
    def _normalize_dtype(dtype: str | torch.dtype) -> str:
        """Normalize supported dtype values to the project's string form."""
        if isinstance(dtype, torch.dtype):
            return str(dtype).removeprefix("torch.")

        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be a str or torch.dtype, got {type(dtype)}")

        normalized_dtype = dtype.removeprefix("torch.")
        if not hasattr(torch, normalized_dtype):
            raise ValueError(f"Unsupported dtype: {dtype!r}")

        torch_dtype = getattr(torch, normalized_dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"Unsupported dtype: {dtype!r}")

        return normalized_dtype

    @staticmethod
    def _to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
        """Convert a supported dtype value to torch.dtype."""
        normalized_dtype = TensorMetadata._normalize_dtype(dtype)
        torch_dtype = getattr(torch, normalized_dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        return torch_dtype

    @staticmethod
    def _promote_dtype(dtype1: str | torch.dtype, dtype2: str | torch.dtype) -> str:
        """
        Promote dtypes using PyTorch's promotion rules.
        """
        promoted_dtype = torch.promote_types(
            TensorMetadata._to_torch_dtype(dtype1),
            TensorMetadata._to_torch_dtype(dtype2),
        )
        return TensorMetadata._normalize_dtype(promoted_dtype)

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
        return f"TensorMetadata(shape={self.shape}, dtype={self.dtype}, device={self.device}{grad_str})"
