"""
Abstract base class for bound representations.

Defines the interface that all bound types must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import torch
from plum import dispatch

from .linear_operators import DenseOperator, LinearOperator
from .regions import AbstractRegion, SimpleRegion


class AbstractBounds(ABC):
    """
    Abstract base class for bound representations.

    All bound types (interval, linear, symbolic) must implement this interface.
    Bounds represent constraints on tensor values - typically lower and upper bounds,
    but can be more complex (e.g., affine relaxations).

    Each bounds object carries a reference to the input region, which is needed
    for concretization (converting symbolic/affine bounds to concrete intervals).

    The key operations are:
    - Propagation through operations (add, mul, matmul, etc.)
    - Combination of bounds from different sources
    - Concretization to intervals for local analysis
    """

    def __init__(self, region: AbstractRegion):
        self.region = region

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Get shape of bounded tensor.

        Returns:
            Shape tuple
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get device of bounds.

        Returns:
            Device where bound tensors are stored
        """
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> AbstractBounds:
        """
        Move bounds to a device.

        Args:
            device: Target device

        Returns:
            New bounds on target device
        """
        pass

    @abstractmethod
    def __getitem__(self, item) -> AbstractBounds:
        """
        Slice/index the bounds.

        Args:
            item: Slice/index specification (e.g., for batch slicing)
        Returns:
            New bounds corresponding to the slice/index
        """
        raise NotImplementedError("Bounds slicing not implemented for this bound type")

    @abstractmethod
    def concretize(self) -> IntervalBounds:
        """
        Concretize bounds to interval bounds.

        This method uses the input region to convert symbolic/affine bounds
        into concrete interval bounds. The default implementation assumes that
        the bounds are already concrete intervals and simply returns them.

        Subclasses with more complex bound types (e.g., linear bounds) should
        override this method to perform the necessary concretization logic.

        Returns:
            IntervalBounds representing the concretized bounds
        """
        raise NotImplementedError("Concretize method must be implemented by subclasses")

    @abstractmethod
    def clone(self) -> AbstractBounds:
        """
        Create a deep copy of these bounds.

        Returns:
            Cloned bounds
        """
        pass


class IntervalBounds(AbstractBounds):
    """
    Interval bounds using simple lower and upper bound tensors.

    This is the simplest form of bounds - just [lower, upper] intervals
    for each element. Propagation uses interval arithmetic rules.

    Attributes:
        lower: Lower bound tensor
        upper: Upper bound tensor
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """
        Initialize interval bounds.

        Args:
            region: Input region (e.g., HyperRectangle)
            lower: Lower bound tensor
            upper: Upper bound tensor

        Raises:
            ValueError: If shapes don't match or bounds are invalid
        """
        if lower.shape != upper.shape:
            raise ValueError(f"Lower and upper bounds must have same shape: {lower.shape} vs {upper.shape}")

        if lower.device != upper.device:
            raise ValueError(f"Lower and upper bounds must be on same device: {lower.device} vs {upper.device}")

        # Check that lower <= upper (allow some numerical tolerance)
        if not torch.all(lower <= upper + 1e-6):
            violations = torch.sum(lower > upper + 1e-6).item()
            raise ValueError(f"Lower bound must be <= upper bound (found {violations} violations)")

        self._lower = lower
        self._upper = upper

    @property
    def lower(self) -> torch.Tensor:
        """Get lower bound tensor."""
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        """Get upper bound tensor."""
        return self._upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self._lower.shape)

    @property
    def device(self) -> torch.device:
        """Get device of bounds."""
        return self._lower.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of bounds."""
        return self._lower.dtype

    def to(self, device: str | torch.device) -> IntervalBounds:
        """
        Move bounds to a device.

        Args:
            device: Target device

        Returns:
            New bounds on target device
        """
        return IntervalBounds(
            lower=self._lower.to(device),
            upper=self._upper.to(device),
        )

    def __iter__(self):
        """Iterate over (lower, upper) to support tuple unpacking."""
        yield self._lower
        yield self._upper

    def __getitem__(self, item) -> IntervalBounds:
        """
        Slice/index the bounds.

        Args:
            item: Slice/index specification (e.g., for batch slicing)
        Returns:
            New bounds corresponding to the slice/index
        """
        return IntervalBounds(
            lower=self._lower[item],
            upper=self._upper[item],
        )

    def concretize(self) -> IntervalBounds:
        """
        Concretize interval bounds to get lower and upper tensors.

        For interval bounds, this simply returns the lower and upper tensors.

        Returns:
            Tuple of (lower, upper) tensors
        """
        return self

    @property
    def width(self) -> torch.Tensor:
        """
        Get interval width (upper - lower).

        Returns:
            Tensor of interval widths
        """
        return self._upper - self._lower

    @property
    def center(self) -> torch.Tensor:
        """
        Get interval center (lower + upper) / 2.

        Returns:
            Tensor of interval centers
        """
        return (self._lower + self._upper) / 2

    @staticmethod
    @dispatch
    def unbounded_like(x: torch.Tensor) -> IntervalBounds:
        """
        Create unbounded interval bounds ([-inf, inf]).

        Args:
            x: Tensor to match shape, device, and dtype

        Returns:
            Unbounded IntervalBounds
        """
        lower = torch.full_like(x, float("-inf"))
        upper = torch.full_like(x, float("inf"))
        return IntervalBounds(lower, upper)

    @staticmethod
    @dispatch
    def unbounded_like(x: IntervalBounds) -> IntervalBounds:  # noqa: F811
        """
        Create unbounded interval bounds ([-inf, inf]) matching another IntervalBounds.

        Args:
            x: IntervalBounds to match shape, device, and dtype

        Returns:
            Unbounded IntervalBounds
        """
        lower = torch.full_like(x.lower, float("-inf"))
        upper = torch.full_like(x.upper, float("inf"))
        return IntervalBounds(lower, upper)

    def clone(self) -> IntervalBounds:
        """
        Create a copy of these bounds.

        Returns:
            New IntervalBounds with cloned tensors
        """
        return IntervalBounds(
            lower=self._lower.clone(),
            upper=self._upper.clone(),
        )


class LinearBounds(AbstractBounds):
    """
    Linear bounds using affine relaxations.

    Represents bounds as affine functions: lower = W_l @ x + b_l, upper = W_u @ x + b_u
    This allows for tighter bounds through linear relaxations of non-linear operations.

    Used in LBP-style bound propagation methods.

    Attributes:
        regions: Input regions defining the domain of each affine term
        linear_lower: Linear coefficients for lower bound (W_l), one tensor per input region
        bias_lower: Bias term for lower bound (b_l)
        linear_upper: Linear coefficients for upper bound (W_u), one tensor per input region
        bias_upper: Bias term for upper bound (b_u)
        input_ids: Optional list of input node IDs that contribute to these bounds.
                   Used for tracking dependencies in multi-input scenarios.
    """

    def __init__(
        self,
        bias_lower: torch.Tensor,
        bias_upper: torch.Tensor,
        linear_lower: torch.Tensor | LinearOperator | Sequence[torch.Tensor | LinearOperator] | None = None,
        linear_upper: torch.Tensor | LinearOperator | Sequence[torch.Tensor | LinearOperator] | None = None,
        regions: SimpleRegion | Sequence[SimpleRegion] | None = None,
        input_ids: int | Sequence[int] | None = None,
    ) -> None:
        """
        Initialize linear bounds.

        Linear coefficients may be provided as raw tensors (auto-wrapped into
        ``DenseOperator``) or as ``LinearOperator`` instances directly.

        Args:
            regions: Input regions, one for each affine coefficient term
            linear_lower: Linear coefficients for lower bound (can be empty for constant bounds)
            bias_lower: Bias for lower bound
            linear_upper: Linear coefficients for upper bound (can be empty for constant bounds)
            bias_upper: Bias for upper bound
            input_ids: Optional list of input node IDs
        """
        normalized_regions = self._normalize_regions(regions)
        normalized_linear_lower = self._normalize_linear_terms(linear_lower, bias_lower.shape, "linear_lower")
        normalized_linear_upper = self._normalize_linear_terms(linear_upper, bias_upper.shape, "linear_upper")
        normalized_input_ids = self._normalize_input_ids(input_ids)

        self._check_uniformity(
            normalized_regions, normalized_linear_lower, normalized_linear_upper, normalized_input_ids
        )

        self._check_shapes(normalized_regions, normalized_linear_lower, bias_lower, normalized_linear_upper, bias_upper)
        self._check_gap(normalized_regions, normalized_linear_lower, bias_lower, normalized_linear_upper, bias_upper)

        self._bias_lower = bias_lower
        self._bias_upper = bias_upper
        self._linear_lower = normalized_linear_lower
        self._linear_upper = normalized_linear_upper
        self._regions = normalized_regions
        self._input_ids = normalized_input_ids

    def has_linear_terms(self) -> bool:
        """Whether these bounds include linear terms (as opposed to being purely constant)."""
        return self._linear_lower is not None and self._linear_upper is not None

    @staticmethod
    def _normalize_regions(regions: SimpleRegion | Sequence[SimpleRegion] | None) -> list[SimpleRegion]:
        if regions is None:
            return []

        if isinstance(regions, SimpleRegion):
            return [regions]

        normalized_regions = list(regions)
        if any(not isinstance(region, SimpleRegion) for region in normalized_regions):
            raise TypeError("regions must contain only SimpleRegion instances")
        return normalized_regions

    @staticmethod
    def _normalize_linear_terms(
        linear_terms: torch.Tensor | LinearOperator | Sequence[torch.Tensor | LinearOperator] | None,
        bias_shape: torch.Size,
        name: str,
    ) -> list[LinearOperator]:
        if linear_terms is None:
            return []

        if isinstance(linear_terms, (torch.Tensor, LinearOperator)):
            return [LinearBounds._wrap_linear_term(linear_terms, bias_shape, name)]

        normalized_terms: list[LinearOperator] = []
        for entry in linear_terms:
            normalized_terms.append(LinearBounds._wrap_linear_term(entry, bias_shape, name))
        return normalized_terms

    @staticmethod
    def _wrap_linear_term(entry: torch.Tensor | LinearOperator, bias_shape: torch.Size, name: str) -> LinearOperator:
        if isinstance(entry, LinearOperator):
            return entry
        if isinstance(entry, torch.Tensor):
            return DenseOperator(entry, output_shape=bias_shape)
        raise TypeError(f"{name} entries must be torch.Tensor or LinearOperator, got {type(entry).__name__}")

    @staticmethod
    def _normalize_input_ids(input_ids: int | Sequence[int] | None) -> list[int]:
        if input_ids is None:
            input_ids = []

        if isinstance(input_ids, int):
            input_ids = [input_ids]

        normalized_input_ids = list(input_ids)
        if len(set(normalized_input_ids)) != len(normalized_input_ids):
            raise ValueError(f"input_ids must be unique, but got {normalized_input_ids!r}")

        return normalized_input_ids

    @staticmethod
    def _split_region_shape(
        region_shape: torch.Size,
        output_shape: torch.Size,
        linear_input_axes: torch.Size,
    ) -> tuple[torch.Size, torch.Size]:
        """Infer (*batch_dims, *input_dims) from linear/bias ranks."""
        input_ndim = len(linear_input_axes)
        batch_ndim = len(region_shape) - input_ndim

        if batch_ndim < 0:
            raise ValueError(
                f"Region rank {len(region_shape)} is smaller than inferred input rank {input_ndim} "
                f"from linear input axes {tuple(linear_input_axes)}"
            )

        if batch_ndim > len(output_shape):
            raise ValueError(
                f"Inferred batch rank {batch_ndim} exceeds output rank {len(output_shape)} "
                f"for region shape {tuple(region_shape)} and output shape {tuple(output_shape)}"
            )

        return region_shape[:batch_ndim], region_shape[batch_ndim:]

    def _check_shapes(
        self,
        regions: list[SimpleRegion],
        linear_lower: list[LinearOperator],
        bias_lower: torch.Tensor,
        linear_upper: list[LinearOperator],
        bias_upper: torch.Tensor,
    ) -> None:
        if bias_lower.shape != bias_upper.shape:
            raise ValueError(
                f"bias_lower and bias_upper must have the same shape: {bias_lower.shape} vs {bias_upper.shape}"
            )

        for name, ops in (("linear_lower", linear_lower), ("linear_upper", linear_upper)):
            for op, region in zip(ops, regions, strict=True):
                if op.output_shape != bias_lower.shape:
                    raise ValueError(
                        f"{name} output shape must match bias shape: {tuple(op.output_shape)} vs "
                        f"{tuple(bias_lower.shape)}"
                    )

                region_shape = torch.Size(region.shape)
                _, input_shape = self._split_region_shape(region_shape, bias_lower.shape, op.input_shape)
                if op.input_shape != input_shape:
                    raise ValueError(
                        f"{name} input axes must match input shape {tuple(input_shape)} "
                        f"(derived from region shape {tuple(region_shape)} and bias shape {tuple(bias_lower.shape)}), "
                        f"got {tuple(op.input_shape)}"
                    )

    @staticmethod
    def _check_uniformity(
        regions: list[SimpleRegion],
        linear_lower: list[LinearOperator],
        linear_upper: list[LinearOperator],
        input_ids: list[int],
    ) -> None:
        if bool(linear_lower) != bool(linear_upper):
            raise ValueError("linear_lower and linear_upper must either both be provided or both be empty")

        if len(linear_lower) != len(regions):
            raise ValueError(
                f"linear_lower must have the same length as regions: {len(linear_lower)} vs {len(regions)}"
            )

        if len(linear_upper) != len(regions):
            raise ValueError(
                f"linear_upper must have the same length as regions: {len(linear_upper)} vs {len(regions)}"
            )

        if len(input_ids) != len(regions):
            raise ValueError(f"input_ids must have the same length as regions: {len(input_ids)} vs {len(regions)}")

    def _check_gap(
        self,
        regions: list[SimpleRegion],
        linear_lower: list[LinearOperator],
        bias_lower: torch.Tensor,
        linear_upper: list[LinearOperator],
        bias_upper: torch.Tensor,
    ) -> None:
        min_gap = bias_upper - bias_lower

        for region, lower_op, upper_op in zip(regions, linear_lower, linear_upper, strict=True):
            diff_op = upper_op.sub(lower_op)
            min_gap = min_gap + diff_op.concretize_min(region)

        if torch.any(min_gap < -1e-6):
            num_violations = torch.sum(min_gap < -1e-6).item()
            raise ValueError(f"Invalid bounds: upper bound is less than lower bound for {num_violations} outputs")

    @staticmethod
    def combine_linear_terms(
        components: list[tuple[LinearBounds, Literal["lower", "upper"], float]],
    ) -> tuple[list[SimpleRegion], list[LinearOperator], list[int]]:
        """
        Merge affine contributions from multiple ``LinearBounds`` keyed by ``input_id``.

        Returns a triple of ``(regions, operators, input_ids)``. The operators
        carry the accumulated coefficient structure (dense by default).
        """
        merged: dict[int, tuple[SimpleRegion, LinearOperator]] = {}
        ordered_input_ids: list[int] = []

        for bounds, bound_side, scale in components:
            ops = bounds.linear_lowers_op if bound_side == "lower" else bounds.linear_uppers_op
            for input_id, region, op in zip(bounds.input_ids, bounds.regions, ops, strict=True):
                if scale == 1:
                    contribution = op
                elif scale == -1:
                    contribution = op.neg()
                else:
                    scale_tensor = torch.tensor(scale, dtype=op.dtype, device=op.device)
                    contribution = op.scale(scale_tensor)

                if input_id in merged:
                    existing_region, existing_op = merged[input_id]
                    if existing_region.shape != region.shape:
                        raise ValueError(
                            "Cannot merge input_id "
                            f"{input_id}: region shapes differ "
                            f"{existing_region.shape} vs {region.shape}"
                        )
                    merged[input_id] = (existing_region, existing_op.add(contribution))
                else:
                    ordered_input_ids.append(input_id)
                    merged[input_id] = (region, contribution)

        regions = [merged[input_id][0] for input_id in ordered_input_ids]
        linear_terms = [merged[input_id][1] for input_id in ordered_input_ids]
        return regions, linear_terms, ordered_input_ids

    @property
    def regions(self) -> list[SimpleRegion]:
        """Get input regions associated with these bounds."""
        return self._regions.copy()

    @property
    def region(self) -> AbstractRegion:
        """Get the single input region associated with these bounds."""
        if len(self._regions) != 1:
            raise ValueError(f"LinearBounds has {len(self._regions)} regions; use regions instead of region")
        return self._regions[0]

    @property
    def input_ids(self) -> list[int]:
        """Get input IDs associated with the affine terms."""
        return self._input_ids.copy()

    @property
    def input_id(self) -> int:
        """Get the single input ID associated with these bounds."""
        if len(self._input_ids) != 1:
            raise ValueError(f"LinearBounds has {len(self._input_ids)} input IDs; use input_ids instead of input_id")
        return self._input_ids[0]

    @property
    def linear_lowers(self) -> list[torch.Tensor]:
        """Get linear coefficients for the lower bound as dense tensors.

        Each operator is materialized via :meth:`LinearOperator.to_dense`. Use
        :attr:`linear_lowers_op` to access the underlying ``LinearOperator``
        instances directly (avoids materializing structured operators).
        """
        return [op.to_dense().tensor for op in self._linear_lower]

    @property
    def linear_lower(self) -> torch.Tensor | None:
        """Get linear coefficients for the lower bound (dense) in the single-input case."""
        if not self._linear_lower:
            return None
        if len(self._linear_lower) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_lower)} lower coefficient terms; use linear_lowers instead"
            )
        return self._linear_lower[0].to_dense().tensor

    @property
    def linear_lowers_op(self) -> list[LinearOperator]:
        """Get lower-bound coefficients as ``LinearOperator`` instances."""
        return self._linear_lower.copy()

    @property
    def linear_lower_op(self) -> LinearOperator | None:
        """Get lower-bound coefficient ``LinearOperator`` in the single-input case."""
        if not self._linear_lower:
            return None
        if len(self._linear_lower) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_lower)} lower coefficient terms; use linear_lowers_op instead"
            )
        return self._linear_lower[0]

    @property
    def bias_lower(self) -> torch.Tensor:
        """Get bias term for lower bound (b_l)."""
        return self._bias_lower

    @property
    def linear_uppers(self) -> list[torch.Tensor]:
        """Get linear coefficients for the upper bound as dense tensors.

        Each operator is materialized via :meth:`LinearOperator.to_dense`. Use
        :attr:`linear_uppers_op` to access the underlying ``LinearOperator``
        instances directly.
        """
        return [op.to_dense().tensor for op in self._linear_upper]

    @property
    def linear_upper(self) -> torch.Tensor | None:
        """Get linear coefficients for the upper bound (dense) in the single-input case."""
        if not self._linear_upper:
            return None
        if len(self._linear_upper) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_upper)} upper coefficient terms; use linear_uppers instead"
            )
        return self._linear_upper[0].to_dense().tensor

    @property
    def linear_uppers_op(self) -> list[LinearOperator]:
        """Get upper-bound coefficients as ``LinearOperator`` instances."""
        return self._linear_upper.copy()

    @property
    def linear_upper_op(self) -> LinearOperator | None:
        """Get upper-bound coefficient ``LinearOperator`` in the single-input case."""
        if not self._linear_upper:
            return None
        if len(self._linear_upper) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._linear_upper)} upper coefficient terms; use linear_uppers_op instead"
            )
        return self._linear_upper[0]

    @property
    def bias_upper(self) -> torch.Tensor:
        """Get bias term for upper bound (b_u)."""
        return self._bias_upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self.bias_lower.shape)

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        """Get shapes of input tensors corresponding to each region."""
        return [tuple(op.input_shape) for op in self._linear_lower]

    @property
    def input_dim(self) -> int:
        """Get input dimensions of linear bounds."""
        if not self._linear_lower:
            return 0

        return sum(
            torch.Size(
                self._split_region_shape(
                    torch.Size(region.shape),
                    self.bias_lower.shape,
                    op.input_shape,
                )[1]
            ).numel()
            for region, op in zip(self._regions, self._linear_lower, strict=True)
        )

    @staticmethod
    def _move_region(region: SimpleRegion, device: str | torch.device) -> SimpleRegion:
        moved_region = region.to(device)
        if not isinstance(moved_region, SimpleRegion):
            raise TypeError(f"Expected SimpleRegion after to(...), got {type(moved_region).__name__}")
        return moved_region

    @property
    def device(self) -> torch.device:
        """Get device of bounds."""
        return self.bias_lower.device

    def to(self, device: str | torch.device) -> LinearBounds:
        """Move bounds to a device."""
        return LinearBounds(
            regions=[self._move_region(region, device) for region in self._regions],
            linear_lower=[op.to(device) for op in self._linear_lower],
            bias_lower=self.bias_lower.to(device),
            linear_upper=[op.to(device) for op in self._linear_upper],
            bias_upper=self.bias_upper.to(device),
            input_ids=self._input_ids.copy() if self._input_ids else None,
        )

    def __getitem__(self, item) -> LinearBounds:
        """Slice/index the bounds over the output (batch) axes; input axes are preserved."""
        return LinearBounds(
            regions=self._regions,
            linear_lower=[op.getitem_output(item) for op in self._linear_lower],
            bias_lower=self.bias_lower[item],
            linear_upper=[op.getitem_output(item) for op in self._linear_upper],
            bias_upper=self.bias_upper[item],
            input_ids=self._input_ids.copy() if self._input_ids else None,
        )

    def concretize(self) -> IntervalBounds:
        """
        Concretize bounds to interval bounds.

        Evaluates each affine term against its region using the operator's
        structured ``concretize_min`` / ``concretize_max`` and accumulates the
        contributions to the constant bias.

        Returns:
            IntervalBounds representing the concretized bounds
        """
        lower_result = self.bias_lower.clone()
        upper_result = self.bias_upper.clone()

        for region, lower_op in zip(self._regions, self._linear_lower, strict=True):
            lower_result = lower_result + lower_op.concretize_min(region)

        for region, upper_op in zip(self._regions, self._linear_upper, strict=True):
            upper_result = upper_result + upper_op.concretize_max(region)

        return IntervalBounds(lower=lower_result, upper=upper_result)

    def clone(self) -> LinearBounds:
        """Create a deep copy."""
        return LinearBounds(
            regions=self._regions,
            linear_lower=[op.clone() for op in self._linear_lower],
            bias_lower=self.bias_lower.clone(),
            linear_upper=[op.clone() for op in self._linear_upper],
            bias_upper=self.bias_upper.clone(),
            input_ids=self._input_ids,
        )
