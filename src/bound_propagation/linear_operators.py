"""Structured linear operators for affine bound coefficients.

:class:`LinearOperator` represents an abstract linear map ``W`` such that a
:class:`LinearBounds` affine term evaluates as ``y = W @ x + b`` without
necessarily materializing ``W`` as a dense ``(*output_shape, *input_shape)``
tensor. This lets structured representations (identity passes, reshapes,
convolutions, pooling) carry their algebraic structure through propagation
instead of immediately densifying.

Shape conventions
-----------------
Mirrors :class:`LinearBounds`::

    output_shape = (*batch_dims, *output_dims)   # matches bias tensor shape
    input_shape  = (*input_dims,)                # trailing axes describing x

A region's tensor shape may have leading batch dims that are absorbed into
``output_shape`` via :func:`_split_region_shape`; the region's trailing axes
are the actual ``input_shape`` that the operator maps from.

Concretization (interval evaluation)
------------------------------------
For a ``HyperRectangle`` region ``[l, u]``, the per-axis sign rule (Mirman
et al. 2018, IBP) gives::

    min_x (W x) = Σ max(W, 0) · l + Σ min(W, 0) · u
    max_x (W x) = Σ max(W, 0) · u + Σ min(W, 0) · l

i.e. positive weights pair with the same-sign endpoint and negative weights
with the opposite endpoint. This is implemented for the dense baseline in
:func:`_hyperrectangle_concretize` and dispatched per-region-type via
:func:`_dense_concretize_min` / ``_max``.

Subclasses
----------
- :class:`DenseOperator` — baseline, wraps the full coefficient tensor.
- :class:`IdentityOperator` — no allocation; ``y = x`` (with batch broadcast).
- :class:`ZeroOperator` — additive identity; produced by
  ``IdentityOperator.clamp_max(0)`` and other absorption paths.
- :class:`ReshapeOperator` — defers shape rearrangement until densification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from plum import dispatch

from .errors import DimensionMismatchError
from .regions import HyperRectangle, SimpleRegion


class LinearOperator(ABC):
    """
    Abstract affine-coefficient tensor for linear bound propagation.

    A ``LinearOperator`` implements the algebra needed by forward-LBP and
    backward-LBP strategies while leaving the storage representation to
    subclasses. Output-axis shape operations (flatten/reshape/transpose/...)
    leave the input axes untouched.
    """

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def output_shape(self) -> torch.Size:
        """Shape of the bias tensor this operator accompanies (``*batch_dims + *output_dims``)."""

    @property
    @abstractmethod
    def input_shape(self) -> torch.Size:
        """Trailing input axes describing the variable ``x``."""

    @property
    def output_ndim(self) -> int:
        return len(self.output_shape)

    @property
    def input_ndim(self) -> int:
        return len(self.input_shape)

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the operator to ``x``.

        ``x`` has shape broadcastable to ``(*batch_dims, *input_shape)`` (where
        ``batch_dims`` is determined by the region). Returns a tensor of shape
        ``output_shape``.
        """

    @abstractmethod
    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint operator to ``y``.

        ``y`` has shape ``(*leading_dims, *output_shape)`` (``leading_dims`` are
        treated as batch). Returns a tensor of shape ``(*leading_dims, *input_shape)``.
        """

    @abstractmethod
    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        """Return ``min_{x ∈ region} W @ x`` with shape ``output_shape``."""

    @abstractmethod
    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        """Return ``max_{x ∈ region} W @ x`` with shape ``output_shape``."""

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    @abstractmethod
    def neg(self) -> LinearOperator: ...

    def scale(self, factor: torch.Tensor) -> LinearOperator:
        """
        Return operator representing elementwise-scaled output ``y' = factor * (W @ x)``.

        ``factor`` must broadcast to ``output_shape``. Default: materialize to
        dense and scale. Subclasses override to keep structured representations
        when possible (e.g. a scaled-conv operator can fold the factor in).
        """
        return self.to_dense().scale(factor)

    def add(self, other: LinearOperator) -> LinearOperator:
        """Return operator representing ``self + other``.

        Default: materialize both sides and add as dense tensors. Subclasses
        override when two structured operators of compatible form can be added
        without materialization (e.g. two convs with matching kernel/stride).

        ``ZeroOperator`` is the additive identity: ``self.add(Zero) == self``
        (subject to broadcast). Detected here so structured ``self`` survives
        without materialization.
        """
        if isinstance(other, ZeroOperator):
            return _add_with_zero(self, other)
        return self.to_dense().add(other.to_dense())

    def sub(self, other: LinearOperator) -> LinearOperator:
        """Return operator representing ``self - other`` (default: ``self + other.neg()``)."""
        return self.add(other.neg())

    # ------------------------------------------------------------------
    # Sign-aware decomposition
    #
    # ``clamp_min`` / ``clamp_max`` are load-bearing: every backward-mode
    # ``F_i`` that performs sign decomposition (auto_LiRPA equation 5)
    # consumes ``A.clamp_min(0)`` and ``A.clamp_max(0)``. Subclasses MUST
    # support these in their native representation when the structured form
    # admits it; the dense fallback below is correct but materializes.
    # ------------------------------------------------------------------

    def clamp_min(self, value: float) -> LinearOperator:
        """Return operator representing entrywise ``max(self, value)``.

        Default: materialize to dense and clamp. Subclasses override when the
        operation can be performed in the native representation (e.g.
        ``DenseOperator`` clamps the stored tensor; ``IdentityOperator``
        with ``value <= 0`` returns ``self``).
        """
        return self.to_dense().clamp_min(value)

    def clamp_max(self, value: float) -> LinearOperator:
        """Return operator representing entrywise ``min(self, value)``.

        Default: materialize to dense and clamp. Subclasses override when the
        operation can be performed in the native representation.
        """
        return self.to_dense().clamp_max(value)

    # ------------------------------------------------------------------
    # Output-axis shape operations (input axes preserved)
    #
    # Defaults here materialize to a :class:`DenseOperator` and delegate. This
    # lets structured subclasses skip
    # implementing every shape op — flatten/transpose/etc. typically break the
    # structural representation anyway, so falling back to dense is the
    # principled default. Subclasses override only the ops they can do
    # natively without materialization.
    # ------------------------------------------------------------------

    def flatten_output(self, start_dim: int, end_dim: int) -> LinearOperator:
        return self.to_dense().flatten_output(start_dim, end_dim)

    def reshape_output(self, shape: tuple[int, ...]) -> LinearOperator:
        return self.to_dense().reshape_output(shape)

    def view_output(self, shape: tuple[int, ...]) -> LinearOperator:
        return self.to_dense().view_output(shape)

    def squeeze_output(self, dim: int | None = None) -> LinearOperator:
        return self.to_dense().squeeze_output(dim)

    def unsqueeze_output(self, dim: int) -> LinearOperator:
        return self.to_dense().unsqueeze_output(dim)

    def transpose_output(self, dim0: int, dim1: int) -> LinearOperator:
        return self.to_dense().transpose_output(dim0, dim1)

    def permute_output(self, dims: tuple[int, ...]) -> LinearOperator:
        return self.to_dense().permute_output(dims)

    def select_output(self, dim: int, index: int) -> LinearOperator:
        return self.to_dense().select_output(dim, index)

    def sum_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> LinearOperator:
        return self.to_dense().sum_output(dim, keepdim)

    def mean_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> LinearOperator:
        return self.to_dense().mean_output(dim, keepdim)

    def getitem_output(self, item) -> LinearOperator:
        """Slice / index over output axes only, preserving input axes."""
        return self.to_dense().getitem_output(item)

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    @abstractmethod
    def to_dense(self) -> DenseOperator:
        """Materialize the operator as a ``DenseOperator``."""

    @abstractmethod
    def to(self, device: str | torch.device) -> LinearOperator: ...

    @abstractmethod
    def clone(self) -> LinearOperator: ...


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _add_with_zero(non_zero: LinearOperator, zero: LinearOperator) -> LinearOperator:
    """Compute ``non_zero + zero`` while preserving structure when possible.

    ``ZeroOperator`` is the additive identity, so the result is ``non_zero``
    when shapes already match. If the broadcast widens the output shape, fall
    back to dense so the wider shape is realized.
    """
    if non_zero.input_shape != zero.input_shape:
        raise DimensionMismatchError(
            f"Cannot add operators with different input shapes: {tuple(non_zero.input_shape)} vs "
            f"{tuple(zero.input_shape)}"
        )
    try:
        merged = torch.broadcast_shapes(non_zero.output_shape, zero.output_shape)
    except RuntimeError as exc:
        raise DimensionMismatchError(
            f"Cannot add operators with incompatible output shapes: {tuple(non_zero.output_shape)} vs "
            f"{tuple(zero.output_shape)}"
        ) from exc
    if torch.Size(merged) == non_zero.output_shape:
        return non_zero
    return non_zero.to_dense().add(zero.to_dense())


# ----------------------------------------------------------------------
# Dense implementation
# ----------------------------------------------------------------------


class DenseOperator(LinearOperator):
    """
    Dense tensor implementation of ``LinearOperator``.

    Wraps a tensor of shape ``(*output_shape, *input_shape)``. All operations
    delegate to ``torch`` primitives on the trailing input axes as singleton
    broadcasts.
    """

    def __init__(self, tensor: torch.Tensor, output_shape: torch.Size | tuple[int, ...]) -> None:
        output_shape = torch.Size(output_shape)
        if tensor.ndim < len(output_shape):
            raise ValueError(
                f"DenseOperator tensor rank {tensor.ndim} is smaller than output rank "
                f"{len(output_shape)} (output_shape={tuple(output_shape)})"
            )
        if tensor.shape[: len(output_shape)] != output_shape:
            raise ValueError(
                f"DenseOperator tensor leading shape {tuple(tensor.shape[: len(output_shape)])} does not "
                f"match output_shape {tuple(output_shape)}"
            )
        self._tensor = tensor
        self._output_shape = output_shape

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """Underlying coefficient tensor of shape ``(*output_shape, *input_shape)``."""
        return self._tensor

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._tensor.shape[len(self._output_shape) :])

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        input_ndim = self.input_ndim
        if x.shape[-input_ndim:] != self.input_shape and input_ndim > 0:
            raise ValueError(
                f"DenseOperator.apply: x.shape trailing {tuple(x.shape[-input_ndim:])} does not match "
                f"input_shape {tuple(self.input_shape)}"
            )
        # Contract the input axes against the trailing axes of the tensor.
        sum_axes = tuple(range(-input_ndim, 0)) if input_ndim > 0 else ()
        if input_ndim == 0:
            return self._tensor
        return (self._tensor * x.reshape(*((1,) * self.output_ndim), *x.shape)).sum(dim=sum_axes)

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        output_ndim = self.output_ndim
        if y.shape[-output_ndim:] != self.output_shape and output_ndim > 0:
            raise ValueError(
                f"DenseOperator.apply_transpose: y.shape trailing {tuple(y.shape[-output_ndim:])} does not "
                f"match output_shape {tuple(self.output_shape)}"
            )
        leading = y.shape[:-output_ndim] if output_ndim > 0 else y.shape
        # Insert singleton input axes on y, then multiply + sum over output axes.
        y_bc = y.reshape(*y.shape, *((1,) * self.input_ndim))
        # tensor shape: (*output_shape, *input_shape); broadcast leading on y.
        contrib = self._tensor * y_bc
        sum_axes = tuple(range(-output_ndim - self.input_ndim, -self.input_ndim)) if output_ndim > 0 else ()
        if output_ndim == 0:
            return contrib
        # After multiplication, shape is (*leading, *output_shape, *input_shape); sum over output axes.
        reduced = contrib.sum(dim=sum_axes)
        return reduced.reshape(*leading, *self.input_shape)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return _dense_concretize_min(region, self._tensor, self._output_shape)  # ty:ignore[invalid-argument-type]

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return _dense_concretize_max(region, self._tensor, self._output_shape)  # ty:ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> DenseOperator:
        return DenseOperator(-self._tensor, self._output_shape)

    def scale(self, factor: torch.Tensor) -> DenseOperator:
        if factor.ndim > self.output_ndim:
            raise ValueError(
                f"scale factor rank {factor.ndim} exceeds output rank {self.output_ndim} "
                f"(output_shape={tuple(self.output_shape)})"
            )
        factor_bc = factor.reshape(factor.shape + (1,) * self.input_ndim)
        return DenseOperator(self._tensor * factor_bc, self._output_shape)

    def add(self, other: LinearOperator) -> LinearOperator:
        if isinstance(other, ZeroOperator):
            return _add_with_zero(self, other)
        if self.input_shape != other.input_shape:
            raise DimensionMismatchError(
                f"Cannot add operators with different input shapes: {tuple(self.input_shape)} vs "
                f"{tuple(other.input_shape)}"
            )
        try:
            merged_output_shape = torch.broadcast_shapes(self.output_shape, other.output_shape)
        except RuntimeError as exc:
            raise DimensionMismatchError(
                f"Cannot add operators with incompatible output shapes: {tuple(self.output_shape)} vs "
                f"{tuple(other.output_shape)}"
            ) from exc
        other_dense = other if isinstance(other, DenseOperator) else other.to_dense()
        # Full tensor sum broadcasts across both output and input axes.
        summed = self._tensor + other_dense.tensor
        return DenseOperator(summed, torch.Size(merged_output_shape))

    def clamp_min(self, value: float) -> DenseOperator:
        return DenseOperator(self._tensor.clamp(min=value), self._output_shape)

    def clamp_max(self, value: float) -> DenseOperator:
        return DenseOperator(self._tensor.clamp(max=value), self._output_shape)

    # ------------------------------------------------------------------
    # Output-axis shape operations
    # ------------------------------------------------------------------

    def flatten_output(self, start_dim: int, end_dim: int) -> DenseOperator:
        start = _normalize_output_dim(start_dim, self.output_ndim, inclusive_end=False)
        end = _normalize_output_dim(end_dim, self.output_ndim, inclusive_end=False)
        if end < start:
            raise ValueError(f"flatten_output end_dim {end} must be >= start_dim {start}")
        new_tensor = self._tensor.flatten(start, end)
        collapsed_size = 1
        for size in self._output_shape[start : end + 1]:
            collapsed_size *= size
        new_output_shape = torch.Size((*self._output_shape[:start], collapsed_size, *self._output_shape[end + 1 :]))
        return DenseOperator(new_tensor, new_output_shape)

    def reshape_output(self, shape: tuple[int, ...]) -> DenseOperator:
        new_tensor = self._tensor.reshape((*shape, *self.input_shape))
        return DenseOperator(new_tensor, torch.Size(shape))

    def view_output(self, shape: tuple[int, ...]) -> DenseOperator:
        new_tensor = self._tensor.view((*shape, *self.input_shape))
        return DenseOperator(new_tensor, torch.Size(shape))

    def squeeze_output(self, dim: int | None = None) -> DenseOperator:
        if dim is None:
            new_output_shape = torch.Size(size for size in self._output_shape if size != 1)
            return self.reshape_output(tuple(new_output_shape))
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=False)
        if self._output_shape[dim] != 1:
            # Match torch semantics: squeeze on non-1 dim is a no-op.
            return self.clone()
        new_tensor = self._tensor.squeeze(dim)
        new_output_shape = torch.Size((*self._output_shape[:dim], *self._output_shape[dim + 1 :]))
        return DenseOperator(new_tensor, new_output_shape)

    def unsqueeze_output(self, dim: int) -> DenseOperator:
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=True)
        new_tensor = self._tensor.unsqueeze(dim)
        new_output_shape = torch.Size((*self._output_shape[:dim], 1, *self._output_shape[dim:]))
        return DenseOperator(new_tensor, new_output_shape)

    def transpose_output(self, dim0: int, dim1: int) -> DenseOperator:
        dim0 = _normalize_output_dim(dim0, self.output_ndim, inclusive_end=False)
        dim1 = _normalize_output_dim(dim1, self.output_ndim, inclusive_end=False)
        new_tensor = self._tensor.transpose(dim0, dim1)
        new_list = list(self._output_shape)
        new_list[dim0], new_list[dim1] = new_list[dim1], new_list[dim0]
        return DenseOperator(new_tensor, torch.Size(new_list))

    def permute_output(self, dims: tuple[int, ...]) -> DenseOperator:
        if len(dims) != self.output_ndim:
            raise ValueError(f"permute_output expects {self.output_ndim} dims, got {len(dims)}")
        normalized = tuple(_normalize_output_dim(d, self.output_ndim, inclusive_end=False) for d in dims)
        if sorted(normalized) != list(range(self.output_ndim)):
            raise ValueError(f"invalid permutation: {normalized}")
        full_dims = (*normalized, *range(self.output_ndim, self._tensor.ndim))
        new_tensor = self._tensor.permute(*full_dims)
        new_output_shape = torch.Size(self._output_shape[d] for d in normalized)
        return DenseOperator(new_tensor, new_output_shape)

    def select_output(self, dim: int, index: int) -> DenseOperator:
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=False)
        new_tensor = self._tensor.select(dim, index)
        new_output_shape = torch.Size((*self._output_shape[:dim], *self._output_shape[dim + 1 :]))
        return DenseOperator(new_tensor, new_output_shape)

    def sum_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> DenseOperator:
        normalized_dim = _normalize_reduction_dim(dim, self.output_ndim)
        new_tensor = self._tensor.sum(dim=normalized_dim, keepdim=keepdim)
        new_output_shape = _apply_reduction_to_shape(self._output_shape, normalized_dim, keepdim)
        return DenseOperator(new_tensor, new_output_shape)

    def mean_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> DenseOperator:
        normalized_dim = _normalize_reduction_dim(dim, self.output_ndim)
        new_tensor = self._tensor.mean(dim=normalized_dim, keepdim=keepdim)
        new_output_shape = _apply_reduction_to_shape(self._output_shape, normalized_dim, keepdim)
        return DenseOperator(new_tensor, new_output_shape)

    def getitem_output(self, item) -> DenseOperator:
        # Slice over output axes; extend with full slices for input axes.
        if isinstance(item, tuple) and Ellipsis in item:
            extended = item + (slice(None),) * self.input_ndim
        else:
            extended = item
        sliced = self._tensor[extended]
        # Derive the resulting output shape by slicing a placeholder.
        new_output_shape = torch.Size(sliced.shape[: sliced.ndim - self.input_ndim])
        return DenseOperator(sliced, new_output_shape)

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        return self

    def to(self, device: str | torch.device) -> DenseOperator:
        return DenseOperator(self._tensor.to(device), self._output_shape)

    def clone(self) -> DenseOperator:
        return DenseOperator(self._tensor.clone(), self._output_shape)


# ----------------------------------------------------------------------
# Concretization dispatchers (region-type-based)
# ----------------------------------------------------------------------


@dispatch
def _dense_concretize_min(region: SimpleRegion, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    raise NotImplementedError(f"Concretization is not implemented for region type {type(region).__name__}")


@dispatch
def _dense_concretize_min(  # noqa: F811
    region: HyperRectangle, linear: torch.Tensor, output_shape: torch.Size
) -> torch.Tensor:
    return _hyperrectangle_concretize(region, linear, output_shape, mode="min")


@dispatch
def _dense_concretize_max(region: SimpleRegion, linear: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    raise NotImplementedError(f"Concretization is not implemented for region type {type(region).__name__}")


@dispatch
def _dense_concretize_max(  # noqa: F811
    region: HyperRectangle, linear: torch.Tensor, output_shape: torch.Size
) -> torch.Tensor:
    return _hyperrectangle_concretize(region, linear, output_shape, mode="max")


def _hyperrectangle_concretize(
    region: HyperRectangle, linear: torch.Tensor, output_shape: torch.Size, *, mode: str
) -> torch.Tensor:
    """Per-axis sign-aware min/max of ``W x`` over a box region.

    For ``x ∈ [l, u]``, separability gives
    ``min_x (W x) = Σ_i max(W_i, 0) · l_i + Σ_i min(W_i, 0) · u_i``
    and the symmetric formula for the maximum (l ↔ u). Implemented per
    element via :func:`torch.where` on the sign of the coefficient: positive
    weights pair with the same-sign endpoint, negative weights with the
    opposite endpoint.

    Reduction is over the trailing ``input_ndim`` axes; the leading
    ``output_ndim`` axes (which may include batch dims absorbed from the
    region) are preserved.
    """
    input_lower = region.lower
    input_upper = region.upper

    region_shape = torch.Size(region.shape)
    linear_input_axes = torch.Size(linear.shape[len(output_shape) :])
    batch_shape, input_shape = _split_region_shape(region_shape, output_shape, linear_input_axes)
    input_ndim = len(input_shape)
    output_ndim = len(output_shape)

    if linear_input_axes != input_shape:
        raise ValueError(
            f"linear input axes {tuple(linear_input_axes)} are incompatible with input shape {tuple(input_shape)} "
            f"derived from region shape {tuple(region_shape)}"
        )

    expanded_shape = (*batch_shape, *([1] * (output_ndim - len(batch_shape))), *input_shape)
    expanded_lower = input_lower.reshape(expanded_shape)
    expanded_upper = input_upper.reshape(expanded_shape)
    if mode == "min":
        # min over [l, u] of w·x: take l where w>0, u where w<=0.
        contributions = torch.where(linear > 0, linear * expanded_lower, linear * expanded_upper)
    elif mode == "max":
        # max over [l, u] of w·x: take u where w>0, l where w<=0.
        contributions = torch.where(linear > 0, linear * expanded_upper, linear * expanded_lower)
    else:
        raise ValueError(f"Invalid concretize mode {mode!r}")

    sum_dims = tuple(range(-input_ndim, 0))
    return contributions.sum(dim=sum_dims) if sum_dims else contributions


def _split_region_shape(
    region_shape: torch.Size, output_shape: torch.Size, linear_input_axes: torch.Size
) -> tuple[torch.Size, torch.Size]:
    """Infer (*batch_dims, *input_dims) of a region given an affine term's input axes."""
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


# ----------------------------------------------------------------------
# Shape helpers
# ----------------------------------------------------------------------


def _normalize_output_dim(dim: int, output_ndim: int, *, inclusive_end: bool) -> int:
    if dim < 0:
        dim += output_ndim + (1 if inclusive_end else 0)
    upper = output_ndim if inclusive_end else output_ndim - 1
    if dim < 0 or dim > upper:
        raise ValueError(f"output dim must be in [0, {upper}], got {dim}")
    return dim


def _normalize_reduction_dim(dim: int | tuple[int, ...] | None, output_ndim: int) -> int | tuple[int, ...]:
    if dim is None:
        return tuple(range(output_ndim))
    if isinstance(dim, int):
        return _normalize_output_dim(dim, output_ndim, inclusive_end=False)
    normalized = tuple(_normalize_output_dim(d, output_ndim, inclusive_end=False) for d in dim)
    return normalized


def _apply_reduction_to_shape(output_shape: torch.Size, dim: int | tuple[int, ...], keepdim: bool) -> torch.Size:
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)
    if keepdim:
        return torch.Size(1 if i in dims else s for i, s in enumerate(output_shape))
    return torch.Size(s for i, s in enumerate(output_shape) if i not in dims)


# ----------------------------------------------------------------------
# Identity operator
# ----------------------------------------------------------------------


class IdentityOperator(LinearOperator):
    """The identity linear map ``y = x`` over a fixed feature shape.

    Used as the placeholder coefficient in forward-LBP (see
    :func:`create_identity_bounds`).

    Shape convention:

    - ``feature_shape`` is the raw input tensor shape (e.g. ``(C, H, W)`` for
      a CNN input).
    - ``output_shape`` equals ``(*batch, *feature_shape)`` where ``batch`` is
      the leading broadcast shape carried by the containing ``LinearBounds``
      (typically all ones). ``input_shape`` equals ``feature_shape``.

    All algebraic operations besides the trivial ones fall back to dense via
    :meth:`to_dense`, which materializes an ``eye(numel(feature_shape))``.
    """

    def __init__(
        self,
        feature_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        batch_shape: tuple[int, ...] = (),
    ) -> None:
        self._feature_shape = torch.Size(feature_shape)
        self._batch_shape = torch.Size(batch_shape)
        self._dtype = dtype
        self._device = device

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size((*self._batch_shape, *self._feature_shape))

    @property
    def input_shape(self) -> torch.Size:
        return self._feature_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def feature_shape(self) -> torch.Size:
        return self._feature_shape

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-len(self._feature_shape) :]) != tuple(self._feature_shape):
            raise ValueError(
                f"IdentityOperator.apply: x trailing shape {tuple(x.shape[-len(self._feature_shape) :])} "
                f"does not match feature_shape {tuple(self._feature_shape)}"
            )
        return x

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        output_ndim = self.output_ndim
        if tuple(y.shape[-output_ndim:]) != tuple(self.output_shape):
            raise ValueError(
                f"IdentityOperator.apply_transpose: y trailing shape {tuple(y.shape[-output_ndim:])} "
                f"does not match output_shape {tuple(self.output_shape)}"
            )
        leading = y.shape[:-output_ndim]
        # Batch dims in output_shape are broadcast size-1, so reshaping them
        # away is free. Any non-size-1 batch dim here would be a constructor
        # violation; treat via reshape (sums would be the principled adjoint
        # but size-1 dims make it a no-op).
        return y.reshape(*leading, *self._feature_shape)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        lower, _ = region.aabb()
        return lower

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        _, upper = region.aabb()
        return upper

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> LinearOperator:
        # No "-identity" structured type; materialize.
        return self.to_dense().neg()

    def clamp_min(self, value: float) -> LinearOperator:
        # Identity entries are 0 or 1; clamping from below at any non-positive
        # value leaves them unchanged.
        if value <= 0:
            return self
        return self.to_dense().clamp_min(value)

    def clamp_max(self, value: float) -> LinearOperator:
        # Symmetric fast path: clamping from above at >=1 leaves identity unchanged.
        if value >= 1:
            return self
        if value == 0:
            # Identity entries clamp to 0 → all-zero map; surface as ``ZeroOperator``
            # so downstream ``add`` short-circuits without dense materialization.
            return ZeroOperator(
                output_shape=self.output_shape,
                input_shape=self._feature_shape,
                dtype=self._dtype,
                device=self._device,
            )
        return self.to_dense().clamp_max(value)

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        numel = 1
        for s in self._feature_shape:
            numel *= int(s)
        eye = torch.eye(numel, dtype=self._dtype, device=self._device)
        tensor = eye.reshape((1,) * len(self._batch_shape) + tuple(self._feature_shape) + tuple(self._feature_shape))
        if self._batch_shape and any(s != 1 for s in self._batch_shape):
            tensor = tensor.expand(*self._batch_shape, *self._feature_shape, *self._feature_shape).contiguous()
        return DenseOperator(tensor, output_shape=self.output_shape)

    def to(self, device: str | torch.device) -> IdentityOperator:
        return IdentityOperator(
            feature_shape=tuple(self._feature_shape),
            dtype=self._dtype,
            device=torch.device(device) if isinstance(device, str) else device,
            batch_shape=tuple(self._batch_shape),
        )

    def clone(self) -> IdentityOperator:
        return IdentityOperator(
            feature_shape=tuple(self._feature_shape),
            dtype=self._dtype,
            device=self._device,
            batch_shape=tuple(self._batch_shape),
        )


# ----------------------------------------------------------------------
# Zero operator
# ----------------------------------------------------------------------


class ZeroOperator(LinearOperator):
    """The zero linear map ``y = 0`` of fixed ``output_shape`` and ``input_shape``.

    Used to represent absent contributions and to short-circuit subgraphs
    whose accumulated A-matrix has been zeroed out (e.g. via
    ``IdentityOperator.clamp_max(0)``). Acts as the additive identity under
    :meth:`add` and as the absorbing element under :meth:`scale` and
    :meth:`neg`.

    No tensor is allocated until :meth:`to_dense` is called.
    """

    def __init__(
        self,
        output_shape: tuple[int, ...] | torch.Size,
        input_shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._output_shape = torch.Size(output_shape)
        self._input_shape = torch.Size(input_shape)
        self._dtype = dtype
        self._device = device

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_ndim > 0 and tuple(x.shape[-self.input_ndim :]) != tuple(self._input_shape):
            raise ValueError(
                f"ZeroOperator.apply: x trailing shape {tuple(x.shape[-self.input_ndim :])} "
                f"does not match input_shape {tuple(self._input_shape)}"
            )
        return torch.zeros(self._output_shape, dtype=self._dtype, device=self._device)

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_ndim > 0 and tuple(y.shape[-self.output_ndim :]) != tuple(self._output_shape):
            raise ValueError(
                f"ZeroOperator.apply_transpose: y trailing shape {tuple(y.shape[-self.output_ndim :])} "
                f"does not match output_shape {tuple(self._output_shape)}"
            )
        leading = y.shape[: y.ndim - self.output_ndim] if self.output_ndim > 0 else y.shape
        return torch.zeros((*leading, *self._input_shape), dtype=self._dtype, device=self._device)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return torch.zeros(self._output_shape, dtype=self._dtype, device=self._device)

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return torch.zeros(self._output_shape, dtype=self._dtype, device=self._device)

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> ZeroOperator:
        return self

    def scale(self, factor: torch.Tensor) -> ZeroOperator:
        return self

    def add(self, other: LinearOperator) -> LinearOperator:
        if isinstance(other, ZeroOperator):
            # Zero + Zero = Zero, taking the broadcast output shape.
            if self._input_shape != other.input_shape:
                raise DimensionMismatchError(
                    f"Cannot add operators with different input shapes: {tuple(self._input_shape)} vs "
                    f"{tuple(other.input_shape)}"
                )
            try:
                merged = torch.broadcast_shapes(self._output_shape, other.output_shape)
            except RuntimeError as exc:
                raise DimensionMismatchError(
                    f"Cannot add operators with incompatible output shapes: {tuple(self._output_shape)} vs "
                    f"{tuple(other.output_shape)}"
                ) from exc
            return ZeroOperator(
                output_shape=tuple(merged), input_shape=self._input_shape, dtype=self._dtype, device=self._device
            )
        # Symmetric to non-Zero ``add(self)``: identity behavior preserves ``other``'s structure.
        return _add_with_zero(other, self)

    def clamp_min(self, value: float) -> LinearOperator:
        # Zero entries are 0; clamp_min(value <= 0) leaves them; clamp_min(value > 0)
        # makes every entry ``value`` (a constant operator).
        if value <= 0:
            return self
        return self.to_dense().clamp_min(value)

    def clamp_max(self, value: float) -> LinearOperator:
        if value >= 0:
            return self
        return self.to_dense().clamp_max(value)

    # ------------------------------------------------------------------
    # Output-axis shape operations — produce another ``ZeroOperator``.
    # ------------------------------------------------------------------

    def reshape_output(self, shape: tuple[int, ...]) -> ZeroOperator:
        new_shape = torch.Size(shape)
        if new_shape.numel() != self._output_shape.numel():
            raise ValueError(
                f"reshape_output cannot change total size: {tuple(self._output_shape)} "
                f"(numel={self._output_shape.numel()}) → {tuple(new_shape)} (numel={new_shape.numel()})"
            )
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def view_output(self, shape: tuple[int, ...]) -> ZeroOperator:
        return self.reshape_output(
            shape
        )  # Same shape checks and new operator as reshape_output; also semantically identical for a zero map.

    def flatten_output(self, start_dim: int, end_dim: int) -> ZeroOperator:
        start = _normalize_output_dim(start_dim, self.output_ndim, inclusive_end=False)
        end = _normalize_output_dim(end_dim, self.output_ndim, inclusive_end=False)
        if end < start:
            raise ValueError(f"flatten_output end_dim {end} must be >= start_dim {start}")
        collapsed = 1
        for s in self._output_shape[start : end + 1]:
            collapsed *= int(s)
        new_shape = (*self._output_shape[:start], collapsed, *self._output_shape[end + 1 :])
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def squeeze_output(self, dim: int | None = None) -> ZeroOperator:
        if dim is None:
            new_shape = tuple(s for s in self._output_shape if s != 1)
            return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=False)
        if self._output_shape[dim] != 1:
            return self.clone()
        new_shape = (*self._output_shape[:dim], *self._output_shape[dim + 1 :])
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def unsqueeze_output(self, dim: int) -> ZeroOperator:
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=True)
        new_shape = (*self._output_shape[:dim], 1, *self._output_shape[dim:])
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def transpose_output(self, dim0: int, dim1: int) -> ZeroOperator:
        dim0 = _normalize_output_dim(dim0, self.output_ndim, inclusive_end=False)
        dim1 = _normalize_output_dim(dim1, self.output_ndim, inclusive_end=False)
        new_list = list(self._output_shape)
        new_list[dim0], new_list[dim1] = new_list[dim1], new_list[dim0]
        return ZeroOperator(torch.Size(new_list), self._input_shape, dtype=self._dtype, device=self._device)

    def permute_output(self, dims: tuple[int, ...]) -> ZeroOperator:
        if len(dims) != self.output_ndim:
            raise ValueError(f"permute_output expects {self.output_ndim} dims, got {len(dims)}")
        normalized = tuple(_normalize_output_dim(d, self.output_ndim, inclusive_end=False) for d in dims)
        if sorted(normalized) != list(range(self.output_ndim)):
            raise ValueError(f"invalid permutation: {normalized}")
        new_shape = torch.Size(self._output_shape[d] for d in normalized)
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def select_output(self, dim: int, index: int) -> ZeroOperator:
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=False)
        new_shape = (*self._output_shape[:dim], *self._output_shape[dim + 1 :])
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def sum_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> ZeroOperator:
        normalized_dim = _normalize_reduction_dim(dim, self.output_ndim)
        new_shape = _apply_reduction_to_shape(self._output_shape, normalized_dim, keepdim)
        return ZeroOperator(new_shape, self._input_shape, dtype=self._dtype, device=self._device)

    def mean_output(self, dim: int | tuple[int, ...] | None, keepdim: bool) -> ZeroOperator:
        return self.sum_output(dim, keepdim)

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        zero_tensor = torch.zeros(
            tuple(self._output_shape) + tuple(self._input_shape),
            dtype=self._dtype,
            device=self._device,
        )
        return DenseOperator(zero_tensor, self._output_shape)

    def to(self, device: str | torch.device) -> ZeroOperator:
        return ZeroOperator(
            output_shape=tuple(self._output_shape),
            input_shape=tuple(self._input_shape),
            dtype=self._dtype,
            device=torch.device(device) if isinstance(device, str) else device,
        )

    def clone(self) -> ZeroOperator:
        return ZeroOperator(
            output_shape=tuple(self._output_shape),
            input_shape=tuple(self._input_shape),
            dtype=self._dtype,
            device=self._device,
        )


# ----------------------------------------------------------------------
# Lazy reshape wrapper
# ----------------------------------------------------------------------


class ReshapeOperator(LinearOperator):
    """Lazy reshape of another operator's output axes.

    Presents an ``output_shape`` whose total size equals the wrapped
    operator's ``output_shape`` total size, without materializing the wrapped
    operator to a dense tensor. This lets structured operators flow through
    ``flatten``/``view``/``reshape``/``squeeze``/``unsqueeze`` in the forward-LBP
    pipeline without paying a materialization cost at the shape-manipulation boundary.

    The wrapper only handles shape transforms that are pure data-order
    permutations of the output axes: reshape-equivalent ops (including
    ``flatten``, ``view``, ``squeeze``, ``unsqueeze``). Transforms that reorder
    values (``transpose``, ``permute``) fall back to the dense default by way
    of :meth:`to_dense`.

    Nesting two ``ReshapeOperator`` s is normalized at construction time into a
    single wrapper around the innermost operator.
    """

    def __init__(self, inner: LinearOperator, output_shape: tuple[int, ...] | torch.Size) -> None:
        new_output_shape = torch.Size(output_shape)
        inner_numel = 1
        for s in inner.output_shape:
            inner_numel *= int(s)
        new_numel = 1
        for s in new_output_shape:
            new_numel *= int(s)
        if inner_numel != new_numel:
            raise ValueError(
                f"ReshapeOperator cannot reshape output_shape {tuple(inner.output_shape)} "
                f"(numel={inner_numel}) to {tuple(new_output_shape)} (numel={new_numel}): "
                "total size must match"
            )
        if isinstance(inner, ReshapeOperator):
            inner = inner._inner
        self._inner = inner
        self._output_shape = new_output_shape

    # ------------------------------------------------------------------
    # Shape / metadata
    # ------------------------------------------------------------------

    @property
    def inner(self) -> LinearOperator:
        """The wrapped operator."""
        return self._inner

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @property
    def input_shape(self) -> torch.Size:
        return self._inner.input_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._inner.dtype

    @property
    def device(self) -> torch.device:
        return self._inner.device

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        inner_result = self._inner.apply(x)
        return self._view_to_output(inner_result)

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        out_ndim = len(self._output_shape)
        if y.ndim < out_ndim or tuple(y.shape[y.ndim - out_ndim :]) != tuple(self._output_shape):
            raise ValueError(
                f"ReshapeOperator.apply_transpose: y.shape trailing "
                f"{tuple(y.shape[y.ndim - out_ndim :])} does not match output_shape "
                f"{tuple(self._output_shape)}"
            )
        leading = y.shape[: y.ndim - out_ndim]
        reshaped = y.reshape((*leading, *self._inner.output_shape))
        return self._inner.apply_transpose(reshaped)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return self._view_to_output(self._inner.concretize_min(region))

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return self._view_to_output(self._inner.concretize_max(region))

    def _view_to_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (via view) the inner result to this operator's output_shape.

        Leading axes where both ``inner.output_shape`` and ``self._output_shape``
        are size 1 are treated as batch placeholders: the corresponding leading
        axes of ``tensor`` (which may have replaced those placeholders with real
        batch sizes) pass through, and only the trailing feature axes are
        reshaped to ``output_shape``'s feature tail.
        """
        inner_shape = self._inner.output_shape
        out_shape = self._output_shape

        n_placeholder = 0
        for inner_dim, out_dim in zip(inner_shape, out_shape, strict=False):
            if inner_dim == 1 and out_dim == 1:
                n_placeholder += 1
            else:
                break

        inner_feature_ndim = len(inner_shape) - n_placeholder
        batch_dims = tuple(tensor.shape[: tensor.ndim - inner_feature_ndim])

        try:
            return tensor.view((*batch_dims, *out_shape[n_placeholder:]))
        except RuntimeError as exc:
            raise DimensionMismatchError(
                f"ReshapeOperator: cannot view inner output shape {tuple(inner_shape)} "
                f"as {tuple(out_shape)}; tensor shape was {tuple(tensor.shape)}"
            ) from exc

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> ReshapeOperator:
        return ReshapeOperator(self._inner.neg(), self._output_shape)

    def scale(self, factor: torch.Tensor) -> LinearOperator:
        if factor.ndim == 0:
            return ReshapeOperator(self._inner.scale(factor), self._output_shape)
        if factor.shape == self._output_shape:
            reshaped_factor = factor.reshape(self._inner.output_shape)
            return ReshapeOperator(self._inner.scale(reshaped_factor), self._output_shape)
        return self.to_dense().scale(factor)

    def add(self, other: LinearOperator) -> LinearOperator:
        if isinstance(other, ZeroOperator):
            return _add_with_zero(self, other)
        if (
            isinstance(other, ReshapeOperator)
            and other._output_shape == self._output_shape
            and other._inner.output_shape == self._inner.output_shape
            and other._inner.input_shape == self._inner.input_shape
        ):
            combined_inner = self._inner.add(other._inner)
            return ReshapeOperator(combined_inner, self._output_shape)
        return self.to_dense().add(other.to_dense())

    def clamp_min(self, value: float) -> LinearOperator:
        # Reshape preserves entries; clamping commutes through.
        return ReshapeOperator(self._inner.clamp_min(value), self._output_shape)

    def clamp_max(self, value: float) -> LinearOperator:
        return ReshapeOperator(self._inner.clamp_max(value), self._output_shape)

    # ------------------------------------------------------------------
    # Output-axis shape operations — compose with the wrapped reshape.
    # Transforms that permute values (transpose/permute) fall back to dense.
    # ------------------------------------------------------------------

    def reshape_output(self, shape: tuple[int, ...]) -> ReshapeOperator:
        return ReshapeOperator(self._inner, tuple(shape))

    def view_output(self, shape: tuple[int, ...]) -> ReshapeOperator:
        return ReshapeOperator(self._inner, tuple(shape))

    def flatten_output(self, start_dim: int, end_dim: int) -> ReshapeOperator:
        start = _normalize_output_dim(start_dim, self.output_ndim, inclusive_end=False)
        end = _normalize_output_dim(end_dim, self.output_ndim, inclusive_end=False)
        if end < start:
            raise ValueError(f"flatten_output end_dim {end} must be >= start_dim {start}")
        collapsed = 1
        for s in self._output_shape[start : end + 1]:
            collapsed *= int(s)
        new_shape = (*self._output_shape[:start], collapsed, *self._output_shape[end + 1 :])
        return ReshapeOperator(self._inner, new_shape)

    def squeeze_output(self, dim: int | None = None) -> ReshapeOperator:
        if dim is None:
            new_shape = tuple(s for s in self._output_shape if s != 1)
            return ReshapeOperator(self._inner, new_shape)
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=False)
        if self._output_shape[dim] != 1:
            return self.clone()
        new_shape = (*self._output_shape[:dim], *self._output_shape[dim + 1 :])
        return ReshapeOperator(self._inner, new_shape)

    def unsqueeze_output(self, dim: int) -> ReshapeOperator:
        dim = _normalize_output_dim(dim, self.output_ndim, inclusive_end=True)
        new_shape = (*self._output_shape[:dim], 1, *self._output_shape[dim:])
        return ReshapeOperator(self._inner, new_shape)

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        return self._inner.to_dense().reshape_output(tuple(self._output_shape))

    def to(self, device: str | torch.device) -> ReshapeOperator:
        return ReshapeOperator(self._inner.to(device), self._output_shape)

    def clone(self) -> ReshapeOperator:
        return ReshapeOperator(self._inner.clone(), self._output_shape)
