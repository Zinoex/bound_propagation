"""Bound representations used during propagation.

This module defines the two concrete bound types — :class:`IntervalBounds`
(``[lower, upper]`` boxes) and :class:`LinearBounds` (per-input affine
relaxations ``W_l x + b_l <= y <= W_u x + b_u``) — and their shared
:class:`AbstractBounds` interface.

Invariants
----------
**Identity-by-``is`` for exactness.** :attr:`LinearCoefficient.is_exact` and
:attr:`LinearBounds.is_exact` use Python ``is`` (not value equality) on the
``lower`` / ``upper`` operators and biases. Construction sites that produce
purely-affine bounds must reuse the **same Python object** for both sides:

- Affine forward strategies in ``propagation/forward_lbp/linear.py``
  (``ForwardLBPLinear``, ``ForwardLBPAdd``, ``ForwardLBPSub``, ``ForwardLBPNeg``).
- Identity helpers in ``propagation/forward_lbp/utils.py``
  (``create_identity_bounds`` ties ``lower`` and ``upper`` to the same operator).
- Backward shape Identity branches in ``propagation/backward_lbp/shape.py``
  (the ``(IdentityOperator, IdentityOperator)`` overloads share one operator).

Strategies use ``is_exact`` as a fast path to skip the upper-side computation
when both the accumulated bound and the local relaxation are exact (sigmoid,
tanh, etc. fall back to the full two-sided path).

**Region-shape vs. feature-shape.** A region's tensor shape may include leading
batch dimensions absorbed via :meth:`LinearBounds._split_region_shape`; the
trailing axes are the actual input feature shape that the coefficient operator
maps from. See :mod:`linear_operators` for the matching shape conventions on
:class:`LinearOperator`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from plum import dispatch

from .linear_operators import DenseOperator, LinearOperator
from .regions import AbstractRegion, SimpleRegion


@dataclass(frozen=True)
class LinearCoefficient:
    """Per-input coefficient block in a multi-input linear bound.

    Bundles one input region with its lower / upper coefficient operators.
    The coefficients are *linear* maps from the input's feature axes to the
    bounded value's feature axes — the bias lives at the
    :class:`LinearBounds` level.

    The ``is_exact`` property is ``True`` iff ``lower is upper`` (same Python
    object), which holds for purely affine layers where lower and upper bounds
    coincide. Strategies can use this as a fast path to skip redundant work.

    Attributes
    ----------
    region : SimpleRegion
        The perturbation set on this input.
    lower : LinearOperator
        Lower-bound coefficient ``A_i^L`` mapping input features to output features.
    upper : LinearOperator
        Upper-bound coefficient ``A_i^U`` mapping input features to output features.
    """

    region: SimpleRegion
    lower: LinearOperator
    upper: LinearOperator

    def __post_init__(self) -> None:
        if self.lower.input_shape != self.upper.input_shape:
            raise ValueError(
                f"LinearCoefficient: lower and upper input shapes must match: "
                f"{tuple(self.lower.input_shape)} vs {tuple(self.upper.input_shape)}"
            )
        if self.lower.output_shape != self.upper.output_shape:
            try:
                torch.broadcast_shapes(self.lower.output_shape, self.upper.output_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"LinearCoefficient: lower and upper output shapes must broadcast: "
                    f"{tuple(self.lower.output_shape)} vs {tuple(self.upper.output_shape)}"
                ) from exc

    @property
    def is_exact(self) -> bool:
        """Whether ``lower is upper`` — set by affine ops where bounds coincide."""
        return self.lower is self.upper


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

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, *, batch_ndim: int = 0) -> None:
        """
        Initialize interval bounds.

        Args:
            lower: Lower bound tensor of shape ``(*batch_dims, *feature_dims)``.
            upper: Upper bound tensor, same shape as ``lower``.
            batch_ndim: Number of leading batch dimensions. Must satisfy
                ``0 <= batch_ndim <= lower.ndim``. Defaults to ``0`` so legacy
                callers that do not yet thread batch information still work.

        Raises:
            ValueError: If shapes don't match, bounds are invalid, or
                ``batch_ndim`` is out of range.
        """
        if lower.shape != upper.shape:
            raise ValueError(f"Lower and upper bounds must have same shape: {lower.shape} vs {upper.shape}")

        if lower.device != upper.device:
            raise ValueError(f"Lower and upper bounds must be on same device: {lower.device} vs {upper.device}")

        if batch_ndim < 0 or batch_ndim > lower.ndim:
            raise ValueError(
                f"batch_ndim must be in [0, {lower.ndim}] for tensor shape {tuple(lower.shape)}, got {batch_ndim}"
            )

        # Check that lower <= upper (allow some numerical tolerance)
        if not torch.all(lower <= upper + 1e-6):
            violations = torch.sum(lower > upper + 1e-6).item()
            raise ValueError(f"Lower bound must be <= upper bound (found {violations} violations)")

        self._lower = lower
        self._upper = upper
        self._batch_ndim = batch_ndim

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

    @property
    def batch_ndim(self) -> int:
        """Number of leading batch dimensions. ``0`` means every dim is a feature dim."""
        return self._batch_ndim

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Trailing feature dimensions — ``shape[batch_ndim:]``."""
        return tuple(self._lower.shape[self._batch_ndim :])

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
            batch_ndim=self._batch_ndim,
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
        sliced_lower = self._lower[item]
        sliced_upper = self._upper[item]
        new_batch_ndim = min(self._batch_ndim, sliced_lower.ndim)
        return IntervalBounds(
            lower=sliced_lower,
            upper=sliced_upper,
            batch_ndim=new_batch_ndim,
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
        return IntervalBounds(lower, upper, batch_ndim=x.batch_ndim)

    def clone(self) -> IntervalBounds:
        """
        Create a copy of these bounds.

        Returns:
            New IntervalBounds with cloned tensors
        """
        return IntervalBounds(
            lower=self._lower.clone(),
            upper=self._upper.clone(),
            batch_ndim=self._batch_ndim,
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
        *,
        coefficients: Mapping[int, LinearCoefficient] | None = None,
        batch_ndim: int = 0,
    ) -> None:
        """
        Initialize linear bounds.

        There are two equivalent construction paths:

        - **Dict path** (preferred): pass ``coefficients={input_id: LinearCoefficient(...)}``
          to provide pre-built per-input blocks directly. ``linear_lower``,
          ``linear_upper``, ``regions``, and ``input_ids`` must be ``None``.
        - **Parallel-list path** (legacy / convenience): pass ``regions``,
          ``linear_lower``, ``linear_upper``, and ``input_ids`` as parallel
          sequences. Linear coefficients may be raw tensors (auto-wrapped into
          ``DenseOperator``) or ``LinearOperator`` instances.

        Internally, both paths produce a ``dict[input_id, LinearCoefficient]``
        used by all accessors and downstream methods.

        Args:
            bias_lower: Bias for lower bound, shape ``(*batch_dims, *feature_dims)``.
            bias_upper: Bias for upper bound, same shape as ``bias_lower``.
            linear_lower: Linear coefficients for lower bound (parallel-list path).
            linear_upper: Linear coefficients for upper bound (parallel-list path).
            regions: Input regions, one per affine term (parallel-list path).
            input_ids: Input node IDs, one per affine term (parallel-list path).
            coefficients: Pre-built ``{input_id: LinearCoefficient}`` mapping (dict path).
            batch_ndim: Number of leading batch dimensions of ``bias_lower``/``bias_upper``.
                Must satisfy ``0 <= batch_ndim <= bias_lower.ndim``. Defaults to ``0``.
        """
        if coefficients is not None:
            if any(arg is not None for arg in (linear_lower, linear_upper, regions, input_ids)):
                raise ValueError(
                    "Pass either the dict path (coefficients=...) or the parallel-list path "
                    "(linear_lower/linear_upper/regions/input_ids), not both."
                )
            normalized_coefficients = self._normalize_coefficients_dict(coefficients)
        else:
            normalized_regions = self._normalize_regions(regions)
            normalized_linear_lower = self._normalize_linear_terms(linear_lower, bias_lower.shape, "linear_lower")
            normalized_linear_upper = self._normalize_linear_terms(linear_upper, bias_upper.shape, "linear_upper")
            normalized_input_ids = self._normalize_input_ids(input_ids)

            self._check_uniformity(
                normalized_regions, normalized_linear_lower, normalized_linear_upper, normalized_input_ids
            )

            normalized_coefficients = {
                input_id: LinearCoefficient(region=region, lower=lower, upper=upper)
                for input_id, region, lower, upper in zip(
                    normalized_input_ids,
                    normalized_regions,
                    normalized_linear_lower,
                    normalized_linear_upper,
                    strict=True,
                )
            }

        self._validate(normalized_coefficients, bias_lower, bias_upper, batch_ndim)

        self._bias_lower = bias_lower
        self._bias_upper = bias_upper
        self._coefficients: dict[int, LinearCoefficient] = normalized_coefficients
        self._batch_ndim = batch_ndim

    def has_linear_terms(self) -> bool:
        """Whether these bounds include linear terms (as opposed to being purely constant)."""
        return bool(self._coefficients)

    @staticmethod
    def _normalize_coefficients_dict(
        coefficients: Mapping[int, LinearCoefficient],
    ) -> dict[int, LinearCoefficient]:
        normalized: dict[int, LinearCoefficient] = {}
        for input_id, coeff in coefficients.items():
            if not isinstance(input_id, int):
                raise TypeError(f"coefficients keys must be int input ids, got {type(input_id).__name__}")
            if not isinstance(coeff, LinearCoefficient):
                raise TypeError(f"coefficients values must be LinearCoefficient instances, got {type(coeff).__name__}")
            normalized[input_id] = coeff
        return normalized

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

    @staticmethod
    def _check_uniformity(
        regions: list[SimpleRegion],
        linear_lower: list[LinearOperator],
        linear_upper: list[LinearOperator],
        input_ids: list[int],
    ) -> None:
        """Validate the parallel-list constructor path before lifting into the dict."""
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

    def _validate(
        self,
        coefficients: dict[int, LinearCoefficient],
        bias_lower: torch.Tensor,
        bias_upper: torch.Tensor,
        batch_ndim: int,
    ) -> None:
        """Run all post-normalization validation in one place.

        Three checks, in order:

        1. ``batch_ndim`` is in ``[0, bias_lower.ndim]``.
        2. Bias shapes match; each coefficient's output shape broadcasts with
           the bias and its input axes match the region's trailing axes.
        3. **Soundness**: ``upper(x) >= lower(x)`` everywhere on the regions,
           computed as ``min_x (b_u - b_l + Σ_i (A_i^U - A_i^L) x_i)``.

        Exact coefficients (``lower is upper``) contribute zero to the gap and
        are skipped. The ``1e-6`` slack on the gap absorbs accumulated float
        error from per-input ``concretize_min`` calls; it is not a tolerance
        for user-supplied data.
        """
        if batch_ndim < 0 or batch_ndim > bias_lower.ndim:
            raise ValueError(
                f"batch_ndim must be in [0, {bias_lower.ndim}] for bias shape {tuple(bias_lower.shape)}, "
                f"got {batch_ndim}"
            )

        if bias_lower.shape != bias_upper.shape:
            raise ValueError(
                f"bias_lower and bias_upper must have the same shape: {bias_lower.shape} vs {bias_upper.shape}"
            )

        for input_id, coeff in coefficients.items():
            for name, op in (("lower", coeff.lower), ("upper", coeff.upper)):
                # Operator output_shape only needs to be broadcast-compatible
                # with bias shape — they represent the same abstract output
                # axes but different allocations may have trailing-1 placeholders
                # the other does not.
                try:
                    torch.broadcast_shapes(op.output_shape, bias_lower.shape)
                except RuntimeError as exc:
                    raise ValueError(
                        f"coefficient[{input_id}].{name} output shape must broadcast with bias shape: "
                        f"{tuple(op.output_shape)} vs {tuple(bias_lower.shape)}"
                    ) from exc

                region_shape = torch.Size(coeff.region.shape)
                _, input_shape = self._split_region_shape(region_shape, bias_lower.shape, op.input_shape)
                if op.input_shape != input_shape:
                    raise ValueError(
                        f"coefficient[{input_id}].{name} input axes must match input shape "
                        f"{tuple(input_shape)} (derived from region shape {tuple(region_shape)} and bias shape "
                        f"{tuple(bias_lower.shape)}), got {tuple(op.input_shape)}"
                    )

        # Soundness check: min over the region of (upper(x) - lower(x)) must be >= -tol.
        min_gap = bias_upper - bias_lower
        for coeff in coefficients.values():
            if coeff.is_exact:
                continue
            diff_op = coeff.upper.sub(coeff.lower)
            min_gap = min_gap + diff_op.concretize_min(coeff.region)

        if torch.any(min_gap < -1e-6):
            num_violations = torch.sum(min_gap < -1e-6).item()
            raise ValueError(f"Invalid bounds: upper bound is less than lower bound for {num_violations} outputs")

    @staticmethod
    def combine_linear_terms(
        components: list[tuple[LinearBounds, Literal["lower", "upper"], float]],
    ) -> tuple[list[SimpleRegion], list[LinearOperator], list[int]]:
        """Merge affine contributions from multiple ``LinearBounds`` keyed by ``input_id``.

        Implements per-input linearity:

        .. math::

            \\sum_k s_k \\cdot (W^{(k)}_i x_i + b^{(k)}_i)
                = \\Big(\\sum_k s_k W^{(k)}_i\\Big) x_i + \\sum_k s_k b^{(k)}_i

        Each component is ``(bounds, side, scale)`` where ``side`` selects
        ``A_i^L`` or ``A_i^U`` of every coefficient and ``scale ∈ {-1, +1, ...}``
        applies a sign / scalar before accumulation. Bias terms are merged
        separately by callers (this method handles only the linear part).

        Regions are merged by **first-encounter order** (the order in which
        each ``input_id`` first appears across ``components``). Two
        contributions for the same ``input_id`` must agree on region shape;
        the regions themselves are taken from the first encounter.

        Returns
        -------
        tuple
            ``(regions, operators, input_ids)`` parallel lists, ordered by
            first encounter. Operators preserve structured types where
            ``add``/``neg``/``scale`` overrides allow it, otherwise fall back
            to dense.
        """
        merged: dict[int, tuple[SimpleRegion, LinearOperator]] = {}

        for bounds, bound_side, scale in components:
            for input_id, coeff in bounds._coefficients.items():
                op = coeff.lower if bound_side == "lower" else coeff.upper
                if scale == 1:
                    contribution = op
                elif scale == -1:
                    contribution = op.neg()
                else:
                    scale_tensor = torch.tensor(scale, dtype=op.dtype, device=op.device)
                    contribution = op.scale(scale_tensor)

                if input_id in merged:
                    existing_region, existing_op = merged[input_id]
                    if existing_region.shape != coeff.region.shape:
                        raise ValueError(
                            "Cannot merge input_id "
                            f"{input_id}: region shapes differ "
                            f"{existing_region.shape} vs {coeff.region.shape}"
                        )
                    merged[input_id] = (existing_region, existing_op.add(contribution))
                else:
                    merged[input_id] = (coeff.region, contribution)

        ordered_input_ids = list(merged.keys())
        regions = [merged[input_id][0] for input_id in ordered_input_ids]
        linear_terms = [merged[input_id][1] for input_id in ordered_input_ids]
        return regions, linear_terms, ordered_input_ids

    @property
    def is_exact(self) -> bool:
        """Whether these bounds are pure-affine — every coefficient and bias coincides.

        ``True`` iff every :class:`LinearCoefficient` has ``lower is upper`` AND
        the lower / upper biases are the same Python object. Strategies use this
        as a fast path to skip the upper-side computation when both the
        accumulated bound and the local relaxation are exact.

        Identity is by Python ``is`` (not value equality) — the affine
        construction sites are responsible for sharing the same operator /
        tensor for both sides, which is cheap and avoids a value comparison
        on every property access.
        """
        if self._bias_lower is not self._bias_upper:
            return False
        return all(c.is_exact for c in self._coefficients.values())

    @property
    def coefficients(self) -> dict[int, LinearCoefficient]:
        """Per-input ``{input_id: LinearCoefficient}`` mapping (defensive copy)."""
        return dict(self._coefficients)

    @property
    def coefficient(self) -> LinearCoefficient:
        """The single ``LinearCoefficient`` in the single-input case."""
        if len(self._coefficients) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._coefficients)} coefficients; use coefficients instead of coefficient"
            )
        return next(iter(self._coefficients.values()))

    @property
    def regions(self) -> list[SimpleRegion]:
        """Get input regions associated with these bounds, in input-id order."""
        return [c.region for c in self._coefficients.values()]

    @property
    def region(self) -> AbstractRegion:
        """Get the single input region associated with these bounds."""
        if len(self._coefficients) != 1:
            raise ValueError(f"LinearBounds has {len(self._coefficients)} regions; use regions instead of region")
        return next(iter(self._coefficients.values())).region

    @property
    def input_ids(self) -> list[int]:
        """Get input IDs associated with the affine terms."""
        return list(self._coefficients.keys())

    @property
    def input_id(self) -> int:
        """Get the single input ID associated with these bounds."""
        if len(self._coefficients) != 1:
            raise ValueError(f"LinearBounds has {len(self._coefficients)} input IDs; use input_ids instead of input_id")
        return next(iter(self._coefficients.keys()))

    @property
    def linear_lowers(self) -> list[torch.Tensor]:
        """Get linear coefficients for the lower bound as dense tensors.

        Each operator is materialized via :meth:`LinearOperator.to_dense`. Use
        :attr:`linear_lowers_op` to access the underlying ``LinearOperator``
        instances directly (avoids materializing structured operators).
        """
        return [c.lower.to_dense().tensor for c in self._coefficients.values()]

    @property
    def linear_lower(self) -> torch.Tensor | None:
        """Single-input convenience: ``linear_lowers[0]`` or ``None`` if empty."""
        if not self._coefficients:
            return None
        if len(self._coefficients) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._coefficients)} lower coefficient terms; use linear_lowers instead"
            )
        return next(iter(self._coefficients.values())).lower.to_dense().tensor

    @property
    def linear_lowers_op(self) -> list[LinearOperator]:
        """Get lower-bound coefficients as ``LinearOperator`` instances."""
        return [c.lower for c in self._coefficients.values()]

    @property
    def linear_lower_op(self) -> LinearOperator | None:
        """Single-input convenience: ``linear_lowers_op[0]`` or ``None`` if empty."""
        if not self._coefficients:
            return None
        if len(self._coefficients) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._coefficients)} lower coefficient terms; use linear_lowers_op instead"
            )
        return next(iter(self._coefficients.values())).lower

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
        return [c.upper.to_dense().tensor for c in self._coefficients.values()]

    @property
    def linear_upper(self) -> torch.Tensor | None:
        """Single-input convenience: ``linear_uppers[0]`` or ``None`` if empty."""
        if not self._coefficients:
            return None
        if len(self._coefficients) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._coefficients)} upper coefficient terms; use linear_uppers instead"
            )
        return next(iter(self._coefficients.values())).upper.to_dense().tensor

    @property
    def linear_uppers_op(self) -> list[LinearOperator]:
        """Get upper-bound coefficients as ``LinearOperator`` instances."""
        return [c.upper for c in self._coefficients.values()]

    @property
    def linear_upper_op(self) -> LinearOperator | None:
        """Single-input convenience: ``linear_uppers_op[0]`` or ``None`` if empty."""
        if not self._coefficients:
            return None
        if len(self._coefficients) != 1:
            raise ValueError(
                f"LinearBounds has {len(self._coefficients)} upper coefficient terms; use linear_uppers_op instead"
            )
        return next(iter(self._coefficients.values())).upper

    @property
    def bias_upper(self) -> torch.Tensor:
        """Get bias term for upper bound (b_u)."""
        return self._bias_upper

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of bounded tensor."""
        return tuple(self.bias_lower.shape)

    @property
    def batch_ndim(self) -> int:
        """Number of leading batch dimensions of the bias / output shape."""
        return self._batch_ndim

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Trailing feature dimensions — ``bias_lower.shape[batch_ndim:]``."""
        return tuple(self.bias_lower.shape[self._batch_ndim :])

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        """Get shapes of input tensors corresponding to each region."""
        return [tuple(c.lower.input_shape) for c in self._coefficients.values()]

    @property
    def input_dim(self) -> int:
        """Get input dimensions of linear bounds."""
        if not self._coefficients:
            return 0

        return sum(
            torch.Size(
                self._split_region_shape(
                    torch.Size(c.region.shape),
                    self.bias_lower.shape,
                    c.lower.input_shape,
                )[1]
            ).numel()
            for c in self._coefficients.values()
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
        moved_coefficients = {
            input_id: LinearCoefficient(
                region=self._move_region(c.region, device),
                lower=c.lower.to(device),
                upper=c.upper.to(device),
            )
            for input_id, c in self._coefficients.items()
        }
        return LinearBounds(
            bias_lower=self.bias_lower.to(device),
            bias_upper=self.bias_upper.to(device),
            coefficients=moved_coefficients,
            batch_ndim=self._batch_ndim,
        )

    def __getitem__(self, item) -> LinearBounds:
        """Slice/index the bounds over the output (batch) axes; input axes are preserved."""
        sliced_bias_lower = self.bias_lower[item]
        new_batch_ndim = min(self._batch_ndim, sliced_bias_lower.ndim)
        sliced_coefficients = {
            input_id: LinearCoefficient(
                region=c.region,
                lower=c.lower.getitem_output(item),
                upper=c.upper.getitem_output(item),
            )
            for input_id, c in self._coefficients.items()
        }
        return LinearBounds(
            bias_lower=sliced_bias_lower,
            bias_upper=self.bias_upper[item],
            coefficients=sliced_coefficients,
            batch_ndim=new_batch_ndim,
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

        for c in self._coefficients.values():
            lower_result = lower_result + c.lower.concretize_min(c.region)
            upper_result = upper_result + c.upper.concretize_max(c.region)

        return IntervalBounds(lower=lower_result, upper=upper_result, batch_ndim=self._batch_ndim)

    def clone(self) -> LinearBounds:
        """Create a deep copy."""
        cloned_coefficients = {
            input_id: LinearCoefficient(
                region=c.region,
                lower=c.lower.clone(),
                upper=c.upper.clone(),
            )
            for input_id, c in self._coefficients.items()
        }
        return LinearBounds(
            bias_lower=self.bias_lower.clone(),
            bias_upper=self.bias_upper.clone(),
            coefficients=cloned_coefficients,
            batch_ndim=self._batch_ndim,
        )
