"""
Structured linear operators for affine bound coefficients.

``LinearOperator`` represents an abstract linear map ``W`` such that a
``LinearBounds`` affine term can be evaluated as ``y = W @ x + b`` without
necessarily materializing ``W`` as a dense ``(*output_shape, *input_shape)``
tensor. This enables structured representations (convolutions, pooling) to
carry their algebraic structure through the bound-propagation pipeline.

Shape conventions mirror ``LinearBounds``:

    output_shape = (*batch_dims, *output_dims)   # matches bias tensor shape
    input_shape  = (*input_dims,)                # trailing axes describing x

A ``DenseOperator`` wraps a tensor of shape ``(*output_shape, *input_shape)``
and is the baseline implementation. Other operators (convolution, pooling)
will be added in later phases and may fall back to ``DenseOperator`` when an
algebraic operation cannot be expressed natively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from plum import dispatch

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
        """
        return self.to_dense().add(other.to_dense())

    def sub(self, other: LinearOperator) -> LinearOperator:
        """Return operator representing ``self - other`` (default: ``self + other.neg()``)."""
        return self.add(other.neg())

    # ------------------------------------------------------------------
    # Composition with a dense linear layer (``nn.Linear`` / ``F.linear`` /
    # constant-matmul). ``weight`` acts on the trailing feature axis of the
    # output space.
    # ------------------------------------------------------------------

    def compose_with_linear_left(self, weight_pos: torch.Tensor, weight_neg: torch.Tensor) -> LinearOperator:
        """Apply a sign-split weight on the trailing feature axis of ``output_shape``.

        Strategies typically use :func:`apply_weight_to_bounds_pair` instead
        of this method directly, since the signed composition needs both
        lower and upper operators jointly. The default implementation raises
        :class:`NotImplementedError`; structured subclasses that can support
        conv-linear composition natively may override.
        """
        raise NotImplementedError(
            "compose_with_linear_left is intentionally unimplemented by default; "
            "strategies should use apply_weight_to_bounds_pair on paired lower/upper operators."
        )

    # ------------------------------------------------------------------
    # Output-axis shape operations (input axes preserved)
    #
    # Defaults here materialize to a :class:`DenseOperator` and delegate. This
    # lets structured subclasses (e.g. :class:`Conv2dOperator`) skip
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
        if self.output_shape != other.output_shape:
            raise ValueError(
                f"Cannot add operators with different output shapes: {tuple(self.output_shape)} vs "
                f"{tuple(other.output_shape)}"
            )
        if self.input_shape != other.input_shape:
            raise ValueError(
                f"Cannot add operators with different input shapes: {tuple(self.input_shape)} vs "
                f"{tuple(other.input_shape)}"
            )
        other_dense = other if isinstance(other, DenseOperator) else other.to_dense()
        return DenseOperator(self._tensor + other_dense.tensor, self._output_shape)

    # ------------------------------------------------------------------
    # Composition with a dense linear layer
    # ------------------------------------------------------------------

    def compose_with_linear_left(self, weight_pos: torch.Tensor, weight_neg: torch.Tensor) -> DenseOperator:
        raise NotImplementedError(
            "compose_with_linear_left is intentionally unimplemented: linear/matmul strategies use "
            "apply_weight_to_bounds on a pair of (lower, upper) operators. See "
            "apply_weight_to_bounds_pair."
        )

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
        new_tensor = self._tensor.reshape(*shape, *self.input_shape)
        return DenseOperator(new_tensor, torch.Size(shape))

    def view_output(self, shape: tuple[int, ...]) -> DenseOperator:
        new_tensor = self._tensor.view(*shape, *self.input_shape)
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
        contributions = torch.where(linear > 0, linear * expanded_lower, linear * expanded_upper)
    elif mode == "max":
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
# Multi-operator combinators
# ----------------------------------------------------------------------

# TODO: update these to avoid materializing to dense when possible;
# e.g. cat_output can often be implemented by concatenating the underlying
# tensors of structured operators without materialization.
# Potentially use @dispatch to implement structured vs dense paths separately.


def cat_output(operators: Sequence[LinearOperator], dim: int) -> LinearOperator:
    """Concatenate operators along an output axis, preserving input axes."""
    if not operators:
        raise ValueError("cat_output requires at least one operator")
    first = operators[0]
    if any(op.input_shape != first.input_shape for op in operators):
        raise ValueError("cat_output requires all operators to share the same input shape")
    if any(op.output_ndim != first.output_ndim for op in operators):
        raise ValueError("cat_output requires all operators to share the same output rank")
    dim = _normalize_output_dim(dim, first.output_ndim, inclusive_end=False)
    # For phase 1 we always materialize to dense; structured subclasses may override.
    tensors = [op.to_dense().tensor for op in operators]
    new_tensor = torch.cat(tensors, dim=dim)
    output_sizes = [op.to_dense().output_shape[dim] for op in operators]
    new_output_shape = torch.Size((*first.output_shape[:dim], sum(output_sizes), *first.output_shape[dim + 1 :]))
    return DenseOperator(new_tensor, new_output_shape)


def stack_output(operators: Sequence[LinearOperator], dim: int) -> LinearOperator:
    """Stack operators along a new output axis."""
    if not operators:
        raise ValueError("stack_output requires at least one operator")
    first = operators[0]
    if any(op.input_shape != first.input_shape for op in operators):
        raise ValueError("stack_output requires all operators to share the same input shape")
    if any(op.output_shape != first.output_shape for op in operators):
        raise ValueError("stack_output requires all operators to share the same output shape")
    dim = _normalize_output_dim(dim, first.output_ndim + 1, inclusive_end=True)
    tensors = [op.to_dense().tensor for op in operators]
    new_tensor = torch.stack(tensors, dim=dim)
    new_output_shape = torch.Size((*first.output_shape[:dim], len(operators), *first.output_shape[dim:]))
    return DenseOperator(new_tensor, new_output_shape)


# ----------------------------------------------------------------------
# Signed composition helper used by forward-LBP linear / matmul strategies.
#
# The typical forward-LBP composition over a linear layer is
#
#     new_lower = weight_pos * op_lower + weight_neg * op_upper
#     new_upper = weight_pos * op_upper + weight_neg * op_lower
#
# where ``weight_pos = weight.clamp(min=0)`` / ``weight_neg = weight.clamp(max=0)``
# act on the trailing feature axis of the output space. Structured operators can
# override this to avoid materializing to dense; the baseline implementation
# falls back to dense einsum.
# ----------------------------------------------------------------------


def apply_weight_to_bounds_pair(
    op_lower: LinearOperator,
    op_upper: LinearOperator,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
    *,
    upper: bool,
    left: bool = True,
) -> LinearOperator:
    """
    Apply a (signed) weight matrix on the feature axis of two paired operators.

    ``left=True`` corresponds to ``y = W @ x`` (nn.Linear / ``F.linear`` /
    ``x @ W`` with W on the right of the reduction, because the convention
    puts the reduction axis last). ``left=False`` corresponds to ``y = x @ W``
    with W treated as a right-multiplied weight whose ``shape[0]`` is the
    reduction axis.

    The default implementation materializes ``op_lower`` / ``op_upper`` to
    dense tensors and performs the einsum in dense space. Structured operators
    can implement their own fast path via duck typing (see Phase 4).
    """
    lower = op_lower.to_dense().tensor
    upper_tensor = op_upper.to_dense().tensor
    output_ndim = op_lower.output_ndim
    output_shape = op_lower.output_shape

    input_axes = lower.shape[output_ndim:]
    batch_shape = lower.shape[: output_ndim - 1]
    feature_dim = lower.shape[output_ndim - 1]

    lower_flat = lower.reshape(*batch_shape, feature_dim, -1)
    upper_flat = upper_tensor.reshape(*batch_shape, feature_dim, -1)

    if weight_pos.shape != weight_neg.shape:
        raise ValueError(
            f"weight_pos and weight_neg must share shape, got {tuple(weight_pos.shape)} vs {tuple(weight_neg.shape)}"
        )

    if weight_pos.ndim == 2:
        if left:
            pos_expr = "ok,...kd->...od"
            out_feature = weight_pos.shape[0]
        else:
            pos_expr = "...kd,ko->...od"
            out_feature = weight_pos.shape[1]
        if upper:
            transformed_flat = torch.einsum(pos_expr, weight_pos, upper_flat) + torch.einsum(
                pos_expr, weight_neg, lower_flat
            )
        else:
            transformed_flat = torch.einsum(pos_expr, weight_pos, lower_flat) + torch.einsum(
                pos_expr, weight_neg, upper_flat
            )
        new_output_shape = torch.Size((*output_shape[:-1], out_feature))
        new_tensor = transformed_flat.reshape(*batch_shape, out_feature, *input_axes)
    elif weight_pos.ndim == 1:
        # 1D weight: dot product reduction; no new feature axis.
        if upper:
            transformed_flat = torch.einsum("k,...kd->...d", weight_pos, upper_flat) + torch.einsum(
                "k,...kd->...d", weight_neg, lower_flat
            )
        else:
            transformed_flat = torch.einsum("k,...kd->...d", weight_pos, lower_flat) + torch.einsum(
                "k,...kd->...d", weight_neg, upper_flat
            )
        new_output_shape = torch.Size(output_shape[:-1])
        new_tensor = transformed_flat.reshape(*batch_shape, *input_axes)
    else:
        raise ValueError(f"weight must be 1D or 2D, got shape {tuple(weight_pos.shape)}")

    # Correct the einsum expression when ``left=False`` + weight is 2D:
    # weight right-multiplies, so the out feature comes from weight.shape[1].
    if not left and weight_pos.ndim == 2:
        # pos_expr was built correctly above ("...kd,ko->...od").
        pass

    return DenseOperator(new_tensor, new_output_shape)


# ----------------------------------------------------------------------
# Identity operator
# ----------------------------------------------------------------------


class IdentityOperator(LinearOperator):
    """The identity linear map ``y = x`` over a fixed feature shape.

    Used as the placeholder coefficient in forward-LBP (see
    :func:`create_identity_bounds`). Its purpose is primarily **type-based
    dispatch**: strategies that can exploit "my input is the raw network
    input" can replace materialization with a structured operator (e.g.
    :class:`ForwardLBPConv2d` emits a :class:`Conv2dOperator` when both of
    its input operators are :class:`IdentityOperator`).

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
        return _identity_concretize(region, self, mode="min")  # ty:ignore[invalid-argument-type]

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return _identity_concretize(region, self, mode="max")  # ty:ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> LinearOperator:
        # No "-identity" structured type; materialize.
        return self.to_dense().neg()

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


@dispatch
def _identity_concretize(region: SimpleRegion, op: IdentityOperator, *, mode: str) -> torch.Tensor:  # noqa: ARG001
    raise NotImplementedError(f"IdentityOperator concretization not implemented for region {type(region).__name__}")


@dispatch
def _identity_concretize(  # noqa: F811
    region: HyperRectangle, op: IdentityOperator, *, mode: str
) -> torch.Tensor:
    source = region.lower if mode == "min" else region.upper
    target = op.output_shape
    if source.shape == target:
        return source
    # Reshape / broadcast — a HyperRectangle's tensor shape may differ from
    # output_shape only in singleton leading dims (from batch_shape inference).
    try:
        return source.reshape(target)
    except RuntimeError:
        return source.expand(target)


# ----------------------------------------------------------------------
# Structured conv operator
# ----------------------------------------------------------------------


class Conv2dOperator(LinearOperator):
    """Structured ``LinearOperator`` for a 2D convolution ``y = conv2d(x, W)``.

    Represents the pure linear map; any bias term belongs to the enclosing
    :class:`LinearBounds`' ``bias_lower`` / ``bias_upper`` tensors.

    Shape conventions:

    - ``input_shape`` is always ``(C_in, H_in, W_in)`` — exactly 3D.
    - ``output_shape`` is ``(*batch, C_out, H_out, W_out)`` — at least 3D;
      any leading batch dims are broadcast-compatible and the conv kernel is
      batch-invariant.

    The operator skips materializing the dense ``(*output_shape, *input_shape)``
    Jacobian. Core ops (``apply``, ``apply_transpose``, ``concretize_{min,max}``,
    ``neg``) stay structural; other operations (shape manipulation, algebraic
    composition with incompatible operators) fall back to dense via ``to_dense``.

    Notes
    -----
    - When the output's trailing spatial shape cannot be inferred from
      ``output_shape`` against the stored hyperparameters, the constructor
      raises :class:`ValueError`.
    - Non-structural composition with, e.g., an ``nn.Linear`` layer will
      materialize to dense at the point of that composition. This is the
      intended memory trade-off: dense is fine at the head of a CNN where
      features have already been downsampled.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        if weight.ndim != 4:
            raise ValueError(
                f"Conv2dOperator weight must be 4D (C_out, C_in/groups, kH, kW), got {tuple(weight.shape)}"
            )
        if len(input_shape) != 3:
            raise ValueError(f"Conv2dOperator input_shape must be 3D (C, H, W), got {tuple(input_shape)}")
        if len(output_shape) < 3:
            raise ValueError(
                f"Conv2dOperator output_shape must be at least 3D (batch..., C_out, H_out, W_out), "
                f"got {tuple(output_shape)}"
            )
        expected_c_out = weight.shape[0]
        if output_shape[-3] != expected_c_out:
            raise ValueError(
                f"Conv2dOperator output_shape[-3] ({output_shape[-3]}) must match weight.shape[0] ({expected_c_out})"
            )

        self._weight = weight
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

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
        return self._weight.dtype

    @property
    def device(self) -> torch.device:
        return self._weight.device

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def stride(self) -> tuple[int, int]:
        return self._stride

    @property
    def padding(self) -> tuple[int, int]:
        return self._padding

    @property
    def dilation(self) -> tuple[int, int]:
        return self._dilation

    @property
    def groups(self) -> int:
        return self._groups

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-3:]) != tuple(self._input_shape):
            raise ValueError(
                f"Conv2dOperator.apply: x trailing shape {tuple(x.shape[-3:])} does not match "
                f"input_shape {tuple(self._input_shape)}"
            )
        leading = x.shape[:-3]
        flat = x.reshape(-1, *self._input_shape)
        out = _conv2d(flat, self._weight, self._stride, self._padding, self._dilation, self._groups)
        return out.reshape(*leading, *out.shape[-3:])

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        output_ndim = self.output_ndim
        if tuple(y.shape[-output_ndim:]) != tuple(self._output_shape):
            raise ValueError(
                f"Conv2dOperator.apply_transpose: y trailing shape {tuple(y.shape[-output_ndim:])} does "
                f"not match output_shape {tuple(self._output_shape)}"
            )
        leading = y.shape[:-output_ndim]

        # Apply conv_transpose2d on the trailing (C_out, H_out, W_out) axes,
        # folding any preceding dims (both caller leading + output-shape batch)
        # into a single flat batch.
        all_flat_batch = y.shape[:-3]
        y_flat = y.reshape(-1, *y.shape[-3:])
        output_padding = _infer_conv_output_padding(
            input_spatial=(int(self._input_shape[-2]), int(self._input_shape[-1])),
            output_spatial=(int(y.shape[-2]), int(y.shape[-1])),
            kernel_size=(int(self._weight.shape[-2]), int(self._weight.shape[-1])),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        x_flat = F.conv_transpose2d(
            y_flat,
            self._weight,
            bias=None,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            output_padding=output_padding,
        )
        x = x_flat.reshape(*all_flat_batch, *self._input_shape)

        # Sum over the batch dims of ``output_shape`` (between caller-leading
        # and the spatial axes) to match :meth:`DenseOperator.apply_transpose`
        # semantics: the adjoint contracts the full output shape.
        batch_of_output = output_ndim - 3
        if batch_of_output > 0:
            reduce_dims = tuple(range(-3 - batch_of_output, -3))
            x = x.sum(dim=reduce_dims)
        return x.reshape(*leading, *self._input_shape)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return _conv2d_concretize(region, self, mode="min")  # ty:ignore[invalid-argument-type]

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return _conv2d_concretize(region, self, mode="max")  # ty:ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> Conv2dOperator:
        return Conv2dOperator(
            weight=-self._weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def add(self, other: LinearOperator) -> LinearOperator:
        """Two Conv2dOperators with identical hyperparameters add kernel-wise;
        otherwise fall back to dense."""
        if (
            isinstance(other, Conv2dOperator)
            and self._stride == other._stride
            and self._padding == other._padding
            and self._dilation == other._dilation
            and self._groups == other._groups
            and tuple(self._input_shape) == tuple(other._input_shape)
            and tuple(self._output_shape) == tuple(other._output_shape)
            and self._weight.shape == other._weight.shape
        ):
            return Conv2dOperator(
                weight=self._weight + other._weight,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
                groups=self._groups,
                input_shape=tuple(self._input_shape),
                output_shape=tuple(self._output_shape),
            )
        return super().add(other)

    def scale(self, factor: torch.Tensor) -> LinearOperator:
        """Per-output-element scaling: return a :class:`ScaledConv2dOperator`.

        ``factor`` must broadcast to ``output_shape``. This keeps the
        conv structure (no dense Jacobian) and enables chained ReLU/sigmoid
        relaxations to stay structural over the same conv kernel.
        """
        factor_bc = _broadcast_factor_to(factor, self._output_shape)
        return ScaledConv2dOperator(
            weight=self._weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            alpha=factor_bc,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        """Materialize as a ``DenseOperator`` by running the conv on an
        input one-hot basis. Memory scales as ``prod(output_shape) *
        prod(input_shape)``; prefer the structural ops for large conv inputs.
        """
        c_in, h_in, w_in = self._input_shape
        numel_in = int(c_in) * int(h_in) * int(w_in)
        basis = torch.eye(numel_in, dtype=self.dtype, device=self.device).reshape(
            numel_in, int(c_in), int(h_in), int(w_in)
        )
        out = _conv2d(basis, self._weight, self._stride, self._padding, self._dilation, self._groups)
        c_out, h_out, w_out = out.shape[-3], out.shape[-2], out.shape[-1]
        # Result currently has shape (numel_in, C_out, H_out, W_out). We want
        # (*output_shape, *input_shape) = (*batch, C_out, H_out, W_out, C_in, H_in, W_in).
        jac = out.permute(1, 2, 3, 0).reshape(c_out, h_out, w_out, int(c_in), int(h_in), int(w_in))

        batch_of_output = tuple(self._output_shape[:-3])
        if batch_of_output:
            # Insert leading singleton dims, then broadcast-expand into the batch shape.
            jac = jac.reshape((1,) * len(batch_of_output) + tuple(jac.shape))
            target = (*batch_of_output, *jac.shape[len(batch_of_output) :])
            jac = jac.expand(*target).contiguous()

        return DenseOperator(jac, output_shape=self._output_shape)

    def to(self, device: str | torch.device) -> Conv2dOperator:
        return Conv2dOperator(
            weight=self._weight.to(device),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def clone(self) -> Conv2dOperator:
        return Conv2dOperator(
            weight=self._weight.clone(),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )


# ----------------------------------------------------------------------
# Scaled conv operator
# ----------------------------------------------------------------------


class ScaledConv2dOperator(LinearOperator):
    """Structured operator ``y = alpha * conv2d(x, W)`` with per-output-element alpha.

    Conceptually a :class:`Conv2dOperator` with an output-space scaling
    factor folded in. Useful for keeping the conv structure through one or
    more subsequent element-wise nonlinearities (ReLU/sigmoid/tanh/...) that
    would otherwise force materialization via their per-output-element alpha.

    Chain preservation: a ``ScaledConv2dOperator.scale(factor)`` returns
    another ``ScaledConv2dOperator`` with ``alpha' = alpha * factor``, and
    ``.add(other)`` with another scaled conv over the *same* weight (and
    hyperparameters) returns a scaled conv with ``alpha' = alpha_self +
    alpha_other``. This means a full ``conv → relu → relu → ... → relu``
    chain stays structural — only a layer that touches the underlying
    spatial structure (a second conv, a flatten, a matmul) triggers
    materialization.

    Shape / math:

    - ``input_shape = (C_in, H_in, W_in)`` — exactly 3D.
    - ``output_shape = (*batch, C_out, H_out, W_out)`` — at least 3D.
    - ``alpha`` has shape ``output_shape`` (per-output-element scaling).

    The weight represents the pure linear conv map; bias still lives in the
    enclosing :class:`LinearBounds`' ``bias_lower`` / ``bias_upper`` tensors.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        alpha: torch.Tensor,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        if weight.ndim != 4:
            raise ValueError(
                f"ScaledConv2dOperator weight must be 4D (C_out, C_in/groups, kH, kW), got {tuple(weight.shape)}"
            )
        if len(input_shape) != 3:
            raise ValueError(f"ScaledConv2dOperator input_shape must be 3D (C, H, W), got {tuple(input_shape)}")
        if len(output_shape) < 3:
            raise ValueError(
                f"ScaledConv2dOperator output_shape must be at least 3D (batch..., C_out, H_out, W_out), "
                f"got {tuple(output_shape)}"
            )
        if output_shape[-3] != weight.shape[0]:
            raise ValueError(
                f"ScaledConv2dOperator output_shape[-3] ({output_shape[-3]}) must match weight.shape[0] "
                f"({weight.shape[0]})"
            )
        if tuple(alpha.shape) != tuple(output_shape):
            raise ValueError(
                f"ScaledConv2dOperator alpha.shape {tuple(alpha.shape)} must equal output_shape {tuple(output_shape)}"
            )

        self._weight = weight
        self._alpha = alpha
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

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
        return self._weight.dtype

    @property
    def device(self) -> torch.device:
        return self._weight.device

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    @property
    def stride(self) -> tuple[int, int]:
        return self._stride

    @property
    def padding(self) -> tuple[int, int]:
        return self._padding

    @property
    def dilation(self) -> tuple[int, int]:
        return self._dilation

    @property
    def groups(self) -> int:
        return self._groups

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-3:]) != tuple(self._input_shape):
            raise ValueError(
                f"ScaledConv2dOperator.apply: x trailing shape {tuple(x.shape[-3:])} does not match "
                f"input_shape {tuple(self._input_shape)}"
            )
        leading = x.shape[:-3]
        flat = x.reshape(-1, *self._input_shape)
        conv_out = _conv2d(flat, self._weight, self._stride, self._padding, self._dilation, self._groups)
        conv_out = conv_out.reshape(*leading, *conv_out.shape[-3:])
        # alpha has shape output_shape; conv_out has shape (*leading, *output_shape[-3:]).
        # Broadcast alpha over any output_shape batch dims that aren't in leading.
        return self._alpha * conv_out

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        output_ndim = self.output_ndim
        if tuple(y.shape[-output_ndim:]) != tuple(self._output_shape):
            raise ValueError(
                f"ScaledConv2dOperator.apply_transpose: y trailing shape {tuple(y.shape[-output_ndim:])} "
                f"does not match output_shape {tuple(self._output_shape)}"
            )
        leading = y.shape[:-output_ndim]

        # Adjoint of y' = alpha * conv(x, W):
        #   <z, alpha * conv(x, W)> = <conv^T(alpha * z), x>
        # Scale by alpha first, then apply conv_transpose.
        scaled = y * self._alpha  # broadcasts over leading dims

        all_flat_batch = scaled.shape[:-3]
        s_flat = scaled.reshape(-1, *scaled.shape[-3:])
        output_padding = _infer_conv_output_padding(
            input_spatial=(int(self._input_shape[-2]), int(self._input_shape[-1])),
            output_spatial=(int(scaled.shape[-2]), int(scaled.shape[-1])),
            kernel_size=(int(self._weight.shape[-2]), int(self._weight.shape[-1])),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        x_flat = F.conv_transpose2d(
            s_flat,
            self._weight,
            bias=None,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            output_padding=output_padding,
        )
        x = x_flat.reshape(*all_flat_batch, *self._input_shape)

        batch_of_output = output_ndim - 3
        if batch_of_output > 0:
            reduce_dims = tuple(range(-3 - batch_of_output, -3))
            x = x.sum(dim=reduce_dims)
        return x.reshape(*leading, *self._input_shape)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return _scaled_conv2d_concretize(region, self, mode="min")  # ty:ignore[invalid-argument-type]

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return _scaled_conv2d_concretize(region, self, mode="max")  # ty:ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> ScaledConv2dOperator:
        return ScaledConv2dOperator(
            weight=self._weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            alpha=-self._alpha,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def scale(self, factor: torch.Tensor) -> LinearOperator:
        """Compose scaling: ``alpha' = alpha * factor`` (broadcast to output_shape)."""
        factor_bc = _broadcast_factor_to(factor, self._output_shape)
        return ScaledConv2dOperator(
            weight=self._weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            alpha=self._alpha * factor_bc,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def add(self, other: LinearOperator) -> LinearOperator:
        """Structural add when ``other`` is a scaled (or plain) conv over the
        same weight and hyperparameters; otherwise fall back to dense.
        """
        if isinstance(other, ScaledConv2dOperator) and self._same_conv_as(
            weight=other._weight,
            stride=other._stride,
            padding=other._padding,
            dilation=other._dilation,
            groups=other._groups,
            input_shape=tuple(other._input_shape),
            output_shape=tuple(other._output_shape),
        ):
            return ScaledConv2dOperator(
                weight=self._weight,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
                groups=self._groups,
                alpha=self._alpha + other._alpha,
                input_shape=tuple(self._input_shape),
                output_shape=tuple(self._output_shape),
            )
        if isinstance(other, Conv2dOperator) and self._same_conv_as(
            weight=other.weight,
            stride=other.stride,
            padding=other.padding,
            dilation=other.dilation,
            groups=other.groups,
            input_shape=tuple(other.input_shape),
            output_shape=tuple(other.output_shape),
        ):
            # Treat ``Conv2dOperator`` as ``ScaledConv2dOperator`` with alpha=1.
            return ScaledConv2dOperator(
                weight=self._weight,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
                groups=self._groups,
                alpha=self._alpha + 1,
                input_shape=tuple(self._input_shape),
                output_shape=tuple(self._output_shape),
            )
        return super().add(other)

    def _same_conv_as(
        self,
        *,
        weight: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> bool:
        if (
            self._stride != stride
            or self._padding != padding
            or self._dilation != dilation
            or self._groups != groups
            or tuple(self._input_shape) != input_shape
            or tuple(self._output_shape) != output_shape
        ):
            return False
        # Weight equality: prefer identity (same tensor object), fall back to
        # value equality. The identity path is hit in the common case where
        # scale/add are chained on the same underlying conv.
        return self._weight is weight or (self._weight.shape == weight.shape and torch.equal(self._weight, weight))

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        """Materialize: dense Jacobian of conv, then scale by alpha broadcast
        over the input axes."""
        base_dense = Conv2dOperator(
            weight=self._weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        ).to_dense()
        # base_dense.tensor: (*output_shape, *input_shape). alpha: output_shape.
        alpha_bc = self._alpha.reshape(self._alpha.shape + (1,) * len(self._input_shape))
        return DenseOperator(base_dense.tensor * alpha_bc, output_shape=self._output_shape)

    def to(self, device: str | torch.device) -> ScaledConv2dOperator:
        return ScaledConv2dOperator(
            weight=self._weight.to(device),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            alpha=self._alpha.to(device),
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def clone(self) -> ScaledConv2dOperator:
        return ScaledConv2dOperator(
            weight=self._weight.clone(),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            alpha=self._alpha.clone(),
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )


@dispatch
def _scaled_conv2d_concretize(  # noqa: ARG001
    region: SimpleRegion, op: ScaledConv2dOperator, *, mode: str
) -> torch.Tensor:
    raise NotImplementedError(
        f"ScaledConv2dOperator concretization not implemented for region type {type(region).__name__}"
    )


@dispatch
def _scaled_conv2d_concretize(  # noqa: F811
    region: HyperRectangle, op: ScaledConv2dOperator, *, mode: str
) -> torch.Tensor:
    # For a per-output-element alpha, sign-decomposition splits by alpha's sign:
    #   min_x (alpha_i * conv(x, W)_i) = alpha_i_pos * conv_min_i + alpha_i_neg * conv_max_i
    # where conv_min/max are the standard sign-decomposed conv bounds.
    lower_in = region.lower
    upper_in = region.upper
    if tuple(lower_in.shape[-3:]) != tuple(op.input_shape):
        raise ValueError(
            f"Region trailing shape {tuple(lower_in.shape[-3:])} does not match input_shape {tuple(op.input_shape)}"
        )

    w_pos = op.weight.clamp(min=0)
    w_neg = op.weight.clamp(max=0)
    leading = lower_in.shape[:-3]
    flat_l = lower_in.reshape(-1, *op.input_shape)
    flat_u = upper_in.reshape(-1, *op.input_shape)

    conv_min_flat = _conv2d(flat_l, w_pos, op._stride, op._padding, op._dilation, op._groups) + _conv2d(
        flat_u, w_neg, op._stride, op._padding, op._dilation, op._groups
    )
    conv_max_flat = _conv2d(flat_u, w_pos, op._stride, op._padding, op._dilation, op._groups) + _conv2d(
        flat_l, w_neg, op._stride, op._padding, op._dilation, op._groups
    )
    conv_min = conv_min_flat.reshape(*leading, *conv_min_flat.shape[-3:])
    conv_max = conv_max_flat.reshape(*leading, *conv_max_flat.shape[-3:])

    # Broadcast to full output_shape.
    if conv_min.shape != op.output_shape:
        conv_min = conv_min.expand(op.output_shape)
        conv_max = conv_max.expand(op.output_shape)

    alpha_pos = op.alpha.clamp(min=0)
    alpha_neg = op.alpha.clamp(max=0)
    if mode == "min":
        return alpha_pos * conv_min + alpha_neg * conv_max
    if mode == "max":
        return alpha_pos * conv_max + alpha_neg * conv_min
    raise ValueError(f"Invalid concretize mode {mode!r}")


def _broadcast_factor_to(factor: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """Broadcast-expand ``factor`` to match ``target`` shape.

    Used when a scaling factor (e.g. from ``ElementwiseForwardRelaxation``) has
    a shape shorter than the target (typically missing leading batch dims),
    or uses size-1 broadcast dims. Returns a contiguous tensor of shape
    ``target``.
    """
    if tuple(factor.shape) == tuple(target):
        return factor
    # Pad leading dims with 1s, then expand.
    padded_shape = (1,) * (len(target) - factor.ndim) + tuple(factor.shape)
    return factor.reshape(padded_shape).expand(target).contiguous()


# ----------------------------------------------------------------------
# Conv2d helpers
# ----------------------------------------------------------------------


def _conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)


def _infer_conv_output_padding(
    input_spatial: tuple[int, int],
    output_spatial: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Derive ``output_padding`` for ``F.conv_transpose2d`` to match ``input_spatial``.

    Mirrors the helper in :mod:`backward_lbp.conv_pool`.
    """
    h_in, w_in = input_spatial
    h_out, w_out = output_spatial
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    h_no_op = (h_out - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1
    w_no_op = (w_out - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1
    op_h = h_in - h_no_op
    op_w = w_in - w_no_op
    if op_h < 0 or op_w < 0 or op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"Cannot infer conv_transpose2d output_padding: "
            f"output_padding=({op_h}, {op_w}) for input_spatial={input_spatial}, "
            f"output_spatial={output_spatial}, stride={stride}"
        )
    return op_h, op_w


@dispatch
def _conv2d_concretize(region: SimpleRegion, conv: Conv2dOperator, *, mode: str) -> torch.Tensor:  # noqa: ARG001
    raise NotImplementedError(
        f"Conv2dOperator concretization is not implemented for region type {type(region).__name__}"
    )


@dispatch
def _conv2d_concretize(  # noqa: F811
    region: HyperRectangle, conv: Conv2dOperator, *, mode: str
) -> torch.Tensor:
    lower_in = region.lower
    upper_in = region.upper

    if tuple(lower_in.shape[-3:]) != tuple(conv.input_shape):
        raise ValueError(
            f"Region trailing shape {tuple(lower_in.shape[-3:])} does not match input_shape {tuple(conv.input_shape)}"
        )

    w_pos = conv.weight.clamp(min=0)
    w_neg = conv.weight.clamp(max=0)

    # Flatten leading batch dims for conv2d.
    leading = lower_in.shape[:-3]
    flat_l = lower_in.reshape(-1, *conv.input_shape)
    flat_u = upper_in.reshape(-1, *conv.input_shape)

    if mode == "min":
        out = _conv2d(flat_l, w_pos, conv.stride, conv.padding, conv.dilation, conv.groups) + _conv2d(
            flat_u, w_neg, conv.stride, conv.padding, conv.dilation, conv.groups
        )
    elif mode == "max":
        out = _conv2d(flat_u, w_pos, conv.stride, conv.padding, conv.dilation, conv.groups) + _conv2d(
            flat_l, w_neg, conv.stride, conv.padding, conv.dilation, conv.groups
        )
    else:
        raise ValueError(f"Invalid concretize mode {mode!r}")

    out = out.reshape(*leading, *out.shape[-3:])
    # Broadcast to output_shape: if output_shape has extra batch dims not in
    # the region, broadcast-expand; if it has fewer, that's a soundness error.
    if out.shape != conv.output_shape:
        try:
            out = out.expand(conv.output_shape)
        except RuntimeError as err:
            raise ValueError(
                f"Conv2dOperator concretization got shape {tuple(out.shape)} but output_shape is "
                f"{tuple(conv.output_shape)}; region shape {tuple(lower_in.shape)} is incompatible"
            ) from err
    return out


# ----------------------------------------------------------------------
# Patch-mode conv operator (position-varying kernel)
# ----------------------------------------------------------------------


class Conv2dPatchOperator(LinearOperator):
    """Linear map with a **position-varying** 2D kernel stored as patches.

    Represents ``y[c_z, h, w] = sum_{c_x, m, n} patch[c_z, h, w, c_x, m, n] *
    x[c_x, h * s + m - p, w * s + n - p]`` (stride=1, dilation=1, groups=1 in
    the current implementation). This generalizes :class:`Conv2dOperator` —
    which has a *single* kernel used at every output position — to the case
    where a composition has introduced position dependence (e.g. a conv after
    a position-dependent scaling from a ReLU relaxation).

    Arises from composing ``conv_k ∘ ScaledConv2dOperator`` via
    :func:`_compose_conv_with_scaled`, where the second conv's kernel absorbs
    the scaled conv's alpha at each output position. The resulting patches
    live on the combined receptive field ``k_combined = k1 + k2 - 1`` (for
    stride-1, dilation-1 convs).

    Shape conventions:

    - ``input_shape = (C_in, H_in, W_in)`` — the original network input shape.
    - ``output_shape = (*batch, C_out, H_out, W_out)``.
    - ``patches`` has shape ``(*output_shape, C_in, k_h, k_w)``.

    Structural algebra:

    - ``scale(factor)`` folds ``factor`` into the patches (per-output-element).
    - ``neg`` negates the patches.
    - ``add(other)`` of two ``Conv2dPatchOperator``s with identical
      hyperparameters and patch shapes sums the patches element-wise; other
      cases fall back to dense.
    - Shape ops (``flatten_output`` etc.) fall back to dense via the ABC default.
    - ``apply_transpose`` falls back to dense.
    """

    def __init__(
        self,
        patches: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        if groups != 1:
            raise NotImplementedError(f"Conv2dPatchOperator currently supports groups=1; got groups={groups}")
        if len(input_shape) != 3:
            raise ValueError(f"Conv2dPatchOperator input_shape must be 3D, got {tuple(input_shape)}")
        if len(output_shape) < 3:
            raise ValueError(f"Conv2dPatchOperator output_shape must be at least 3D, got {tuple(output_shape)}")
        c_in = input_shape[0]
        expected_patches_shape = (*output_shape, c_in, *patches.shape[-2:])
        if tuple(patches.shape) != tuple(expected_patches_shape):
            raise ValueError(
                f"Conv2dPatchOperator patches shape {tuple(patches.shape)} must equal "
                f"(*output_shape, C_in, k_h, k_w) = {tuple(expected_patches_shape)}"
            )

        self._patches = patches
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._input_shape = torch.Size(input_shape)
        self._output_shape = torch.Size(output_shape)

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
        return self._patches.dtype

    @property
    def device(self) -> torch.device:
        return self._patches.device

    @property
    def patches(self) -> torch.Tensor:
        return self._patches

    @property
    def kernel_size(self) -> tuple[int, int]:
        return (int(self._patches.shape[-2]), int(self._patches.shape[-1]))

    @property
    def stride(self) -> tuple[int, int]:
        return self._stride

    @property
    def padding(self) -> tuple[int, int]:
        return self._padding

    @property
    def dilation(self) -> tuple[int, int]:
        return self._dilation

    @property
    def groups(self) -> int:
        return self._groups

    # ------------------------------------------------------------------
    # Core linear-map operations
    # ------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """y[b, c_z, h, w] = sum_{c_x, m, n} patches[*, c_z, h, w, c_x, m, n] * x_unfold[*, c_x, m, n, h, w]."""
        if tuple(x.shape[-3:]) != tuple(self._input_shape):
            raise ValueError(
                f"Conv2dPatchOperator.apply: x trailing shape {tuple(x.shape[-3:])} does not match "
                f"input_shape {tuple(self._input_shape)}"
            )
        c_in, k_h, k_w = int(self._input_shape[0]), *self.kernel_size
        h_out, w_out = int(self._output_shape[-2]), int(self._output_shape[-1])

        leading = x.shape[:-3]
        x_flat = x.reshape(-1, *self._input_shape)
        # Unfold x through the effective receptive field.
        x_unf = F.unfold(
            x_flat,
            kernel_size=(k_h, k_w),
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )
        x_unf = x_unf.reshape(-1, c_in, k_h, k_w, h_out, w_out)
        x_unf = x_unf.reshape(*leading, c_in, k_h, k_w, h_out, w_out)

        # patches: (*output_shape, C_in, k_h, k_w) where output_shape =
        # (*batch_out, C_z, H_out, W_out).  x_unf: (*leading, C_in, k_h, k_w,
        # H_out, W_out). The einsum contracts (C_in, k_h, k_w) and keeps
        # (..., C_z, H_out, W_out).
        # We treat trailing batch of output_shape and leading of x as sharing.
        return torch.einsum("...zhwikl,...iklhw->...zhw", self._patches, x_unf)

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """Structured adjoint via :func:`F.fold`.

        Forward is ``y = einsum("...zhwcmn,...cmnhw->...zhw", patches, x_unfold)``
        where ``x_unfold = F.unfold(x, kernel_size, padding, stride, dilation)``.
        Adjoint: contract ``patches`` against ``y`` on the output dims, then
        ``F.fold`` (the adjoint of ``F.unfold``) back into the input grid —
        which correctly sums overlapping patch contributions.

        Matches :meth:`DenseOperator.apply_transpose` semantics: reduces over
        all of ``output_shape`` (including any batch prefix in the operator's
        output).
        """
        output_ndim = self.output_ndim
        if tuple(y.shape[-output_ndim:]) != tuple(self._output_shape):
            raise ValueError(
                f"Conv2dPatchOperator.apply_transpose: y trailing shape "
                f"{tuple(y.shape[-output_ndim:])} does not match output_shape "
                f"{tuple(self._output_shape)}"
            )
        leading_ndim = y.ndim - output_ndim
        leading_shape = y.shape[:leading_ndim]
        c_in = int(self._input_shape[0])
        k_h, k_w = self.kernel_size
        h_in = int(self._input_shape[-2])
        w_in = int(self._input_shape[-1])
        h_z = int(self._output_shape[-2])
        w_z = int(self._output_shape[-1])

        # Broadcast-multiply patches against y, summing over the part of
        # output_shape before (H_z, W_z) — i.e. *batch_out and C_z.
        y_expanded = y.reshape(y.shape + (1, 1, 1))
        patches_expanded = self._patches.reshape((1,) * leading_ndim + self._patches.shape)
        product = patches_expanded * y_expanded  # (*leading, *output_shape, C_in, k_h, k_w)

        sum_dims = tuple(range(leading_ndim, leading_ndim + output_ndim - 2))
        reduced = product.sum(dim=sum_dims) if sum_dims else product
        # reduced shape: (*leading, H_z, W_z, C_in, k_h, k_w)

        # Permute to (*leading, C_in, k_h, k_w, H_z, W_z) for F.fold.
        perm = list(range(leading_ndim)) + [
            leading_ndim + 2,
            leading_ndim + 3,
            leading_ndim + 4,
            leading_ndim,
            leading_ndim + 1,
        ]
        permuted = reduced.permute(perm)

        fold_input = permuted.reshape(-1, c_in * k_h * k_w, h_z * w_z)
        folded = F.fold(
            fold_input,
            output_size=(h_in, w_in),
            kernel_size=(k_h, k_w),
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
        )
        return folded.reshape(*leading_shape, c_in, h_in, w_in)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return _patch_concretize(region, self, mode="min")  # ty:ignore[invalid-argument-type]

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return _patch_concretize(region, self, mode="max")  # ty:ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def neg(self) -> Conv2dPatchOperator:
        return Conv2dPatchOperator(
            patches=-self._patches,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def scale(self, factor: torch.Tensor) -> LinearOperator:
        factor_bc = _broadcast_factor_to(factor, self._output_shape)
        # Append singleton dims for (C_in, k_h, k_w) to broadcast over patches.
        extra = factor_bc.reshape(factor_bc.shape + (1, 1, 1))
        return Conv2dPatchOperator(
            patches=self._patches * extra,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def add(self, other: LinearOperator) -> LinearOperator:
        if isinstance(other, Conv2dPatchOperator) and self._same_patch_hyperparams(other):
            return Conv2dPatchOperator(
                patches=self._patches + other._patches,
                stride=self._stride,
                padding=self._padding,
                dilation=self._dilation,
                groups=self._groups,
                input_shape=tuple(self._input_shape),
                output_shape=tuple(self._output_shape),
            )
        return super().add(other)

    def _same_patch_hyperparams(self, other: Conv2dPatchOperator) -> bool:
        return (
            self._stride == other._stride
            and self._padding == other._padding
            and self._dilation == other._dilation
            and self._groups == other._groups
            and tuple(self._input_shape) == tuple(other._input_shape)
            and tuple(self._output_shape) == tuple(other._output_shape)
            and self._patches.shape[-2:] == other._patches.shape[-2:]
        )

    # ------------------------------------------------------------------
    # Materialization + housekeeping
    # ------------------------------------------------------------------

    def to_dense(self) -> DenseOperator:
        """Materialize the full ``(*output_shape, *input_shape)`` Jacobian.

        Vectorized ``scatter_add_`` implementation. For each output position
        ``(h_o, w_o)`` and kernel offset ``(m, n)``, the patch entry lands at
        input position ``(r, c) = (h_o*s - p + m*d, w_o*s - p + n*d)``. We
        compute all ``(r, c)`` indices in a single broadcast, mask
        out-of-bounds entries with a boolean ``valid`` mask (so zeroed
        source contributions become no-ops under ``scatter_add_``), then
        scatter along a flattened ``H_in * W_in`` axis.
        """
        c_in, h_in, w_in = (int(s) for s in self._input_shape)
        k_h, k_w = self.kernel_size
        h_out = int(self._output_shape[-2])
        w_out = int(self._output_shape[-1])
        p_h, p_w = self._padding
        s_h, s_w = self._stride
        d_h, d_w = self._dilation
        batch_shape = tuple(self._output_shape[:-3])
        c_out = int(self._output_shape[-3])
        device = self.device
        dtype = self.dtype

        # Per-position row / col indices into the input grid.
        # rows[h_o, m] = h_o * s_h - p_h + m * d_h.
        h_o_arange = torch.arange(h_out, device=device)
        w_o_arange = torch.arange(w_out, device=device)
        m_arange = torch.arange(k_h, device=device)
        n_arange = torch.arange(k_w, device=device)
        rows = h_o_arange.unsqueeze(1) * s_h - p_h + m_arange.unsqueeze(0) * d_h  # (H_o, k_h)
        cols = w_o_arange.unsqueeze(1) * s_w - p_w + n_arange.unsqueeze(0) * d_w  # (W_o, k_w)

        rows_valid = (rows >= 0) & (rows < h_in)  # (H_o, k_h)
        cols_valid = (cols >= 0) & (cols < w_in)  # (W_o, k_w)
        # valid[h_o, w_o, m, n] = rows_valid[h_o, m] & cols_valid[w_o, n]
        valid = rows_valid.unsqueeze(1).unsqueeze(3) & cols_valid.unsqueeze(0).unsqueeze(2)
        # shape: (H_o, W_o, k_h, k_w)

        # Clamp to valid range so indices can always be used with scatter; invalid
        # entries are zeroed in the source tensor so their scatters are no-ops.
        rows_safe = rows.clamp(0, h_in - 1)
        cols_safe = cols.clamp(0, w_in - 1)
        flat_idx = rows_safe.unsqueeze(1).unsqueeze(3) * w_in + cols_safe.unsqueeze(0).unsqueeze(2)
        # shape: (H_o, W_o, k_h, k_w), values in [0, H_in * W_in).

        # Broadcast valid / flat_idx to match ``patches``' shape. Patches:
        # (*batch, C_z, H_o, W_o, C_x, k_h, k_w) — insert singletons for batch,
        # C_z, C_x dimensions.
        bc_shape = (1,) * len(batch_shape) + (1, h_out, w_out, 1, k_h, k_w)
        valid_bc = valid.reshape(bc_shape)
        flat_idx_bc = flat_idx.reshape(bc_shape)

        # Mask source; the clamped flat_idx for invalid entries would otherwise
        # alias a valid entry's destination.
        src = self._patches * valid_bc

        # Flatten kernel dims for a single scatter along the input-spatial axis.
        src_flat = src.reshape(*src.shape[:-2], k_h * k_w)
        idx_flat = flat_idx_bc.reshape(*bc_shape[:-2], k_h * k_w).expand_as(src_flat).contiguous()

        dense_flat = torch.zeros(*src_flat.shape[:-1], h_in * w_in, dtype=dtype, device=device)
        dense_flat.scatter_add_(dim=-1, index=idx_flat, src=src_flat)

        dense = dense_flat.reshape(*dense_flat.shape[:-1], h_in, w_in)
        assert dense.shape == (*batch_shape, c_out, h_out, w_out, c_in, h_in, w_in)
        return DenseOperator(dense, output_shape=self._output_shape)

    def to(self, device: str | torch.device) -> Conv2dPatchOperator:
        return Conv2dPatchOperator(
            patches=self._patches.to(device),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )

    def clone(self) -> Conv2dPatchOperator:
        return Conv2dPatchOperator(
            patches=self._patches.clone(),
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
            input_shape=tuple(self._input_shape),
            output_shape=tuple(self._output_shape),
        )


@dispatch
def _patch_concretize(region: SimpleRegion, op: Conv2dPatchOperator, *, mode: str) -> torch.Tensor:  # noqa: ARG001
    raise NotImplementedError(
        f"Conv2dPatchOperator concretization not implemented for region type {type(region).__name__}"
    )


@dispatch
def _patch_concretize(  # noqa: F811
    region: HyperRectangle, op: Conv2dPatchOperator, *, mode: str
) -> torch.Tensor:
    lower_in = region.lower
    upper_in = region.upper
    if tuple(lower_in.shape[-3:]) != tuple(op.input_shape):
        raise ValueError(
            f"Region trailing shape {tuple(lower_in.shape[-3:])} does not match input_shape {tuple(op.input_shape)}"
        )

    c_in, _, _ = op.input_shape
    k_h, k_w = op.kernel_size
    h_out, w_out = int(op.output_shape[-2]), int(op.output_shape[-1])

    leading = lower_in.shape[:-3]
    l_flat = lower_in.reshape(-1, *op.input_shape)
    u_flat = upper_in.reshape(-1, *op.input_shape)

    l_unf = F.unfold(l_flat, kernel_size=(k_h, k_w), dilation=op._dilation, padding=op.padding, stride=op.stride)
    u_unf = F.unfold(u_flat, kernel_size=(k_h, k_w), dilation=op._dilation, padding=op.padding, stride=op.stride)
    l_unf = l_unf.reshape(*leading, int(c_in), k_h, k_w, h_out, w_out)
    u_unf = u_unf.reshape(*leading, int(c_in), k_h, k_w, h_out, w_out)

    patches = op.patches
    patch_pos = patches.clamp(min=0)
    patch_neg = patches.clamp(max=0)

    if mode == "min":
        out = torch.einsum("...zhwikl,...iklhw->...zhw", patch_pos, l_unf) + torch.einsum(
            "...zhwikl,...iklhw->...zhw", patch_neg, u_unf
        )
    elif mode == "max":
        out = torch.einsum("...zhwikl,...iklhw->...zhw", patch_pos, u_unf) + torch.einsum(
            "...zhwikl,...iklhw->...zhw", patch_neg, l_unf
        )
    else:
        raise ValueError(f"Invalid concretize mode {mode!r}")

    if out.shape != op.output_shape:
        out = out.expand(op.output_shape)
    return out


def _compose_conv_with_scaled(
    scaled_op: ScaledConv2dOperator,
    weight2: torch.Tensor,
    stride2: tuple[int, int],
    padding2: tuple[int, int],
    dilation2: tuple[int, int],
    groups2: int,
    output_shape: tuple[int, ...],
) -> Conv2dPatchOperator:
    """Compose a 2D conv after a :class:`ScaledConv2dOperator` into a
    :class:`Conv2dPatchOperator` that preserves the structure.

    Math (stride=1, dilation=1, groups=1, for both convs):

    Let ``y = alpha * conv(x, W1, p1)`` and ``z = conv(y, W2, p2)``. Then
    ``z[c_z, h, w] = sum_{c_x, m, n} patch[c_z, h, w, c_x, m, n] *
    x[c_x, h + m - p_combined, w + n - p_combined]`` with
    ``k_combined = k1 + k2 - 1`` and ``p_combined = p1 + p2``, and

        patch[c_z, h, w, c_x, m, n] = sum_{c_y, di, dj}
            W2[c_z, c_y, di, dj] * alpha[c_y, h+di-p2, w+dj-p2] * W1[c_y, c_x, m-di, n-dj]

    Implemented by (1) unfolding ``alpha`` through conv₂'s receptive field,
    (2) multiplying by ``W2`` to get an intermediate tensor indexed by
    (c_z, c_y, di, dj, h, w), then (3) applying a 2-D transposed convolution
    against ``W1`` to expand the kernel offsets into the combined receptive
    field (m, n).

    Raises ``NotImplementedError`` when stride/dilation/groups ≠ baseline.
    """
    if scaled_op.stride != (1, 1) or stride2 != (1, 1):
        raise NotImplementedError(
            f"_compose_conv_with_scaled requires stride=(1,1) on both convs; got {scaled_op.stride} and {stride2}"
        )
    if scaled_op.dilation != (1, 1) or dilation2 != (1, 1):
        raise NotImplementedError(
            f"_compose_conv_with_scaled requires dilation=(1,1); got {scaled_op.dilation} and {dilation2}"
        )
    if scaled_op.groups != 1 or groups2 != 1:
        raise NotImplementedError(f"_compose_conv_with_scaled requires groups=1; got {scaled_op.groups} and {groups2}")
    if weight2.ndim != 4:
        raise ValueError(f"weight2 must be 4D (C_z, C_y, k2h, k2w), got {tuple(weight2.shape)}")

    w1 = scaled_op.weight  # (C_y, C_x, k1h, k1w)
    alpha = scaled_op.alpha  # shape = scaled_op.output_shape = (*batch, C_y, H_y, W_y)
    c_z, c_y_w2, k2h, k2w = weight2.shape
    c_y_w1, c_x, k1h, k1w = w1.shape
    if c_y_w2 != c_y_w1:
        raise ValueError(f"Intermediate channel mismatch: weight2 has C_y={c_y_w2}, scaled_op.weight has C_y={c_y_w1}")
    c_y = c_y_w1
    p1h, p1w = scaled_op.padding
    p2h, p2w = padding2

    h_z = int(output_shape[-2])
    w_z = int(output_shape[-1])

    # Flatten any leading batch dims in alpha so F.unfold treats them as batch.
    batch_shape = alpha.shape[:-3]
    alpha_flat = alpha.reshape(-1, c_y, alpha.shape[-2], alpha.shape[-1])
    B = alpha_flat.shape[0]

    # Unfold α through conv₂'s kernel: (B, C_y*k2h*k2w, H_z*W_z) → (B, C_y, k2h, k2w, H_z, W_z).
    alpha_unf = F.unfold(alpha_flat, kernel_size=(k2h, k2w), padding=(p2h, p2w), stride=1)
    alpha_unf = alpha_unf.reshape(B, c_y, k2h, k2w, h_z, w_z)

    # Multiply by W2. Result: (B, C_z, C_y, k2h, k2w, H_z, W_z).
    a = weight2.reshape(1, c_z, c_y, k2h, k2w, 1, 1) * alpha_unf.reshape(B, 1, c_y, k2h, k2w, h_z, w_z)

    # Reshape to (B*C_z*H_z*W_z, C_y, k2h, k2w) for F.conv_transpose2d.
    a_perm = a.permute(0, 1, 5, 6, 2, 3, 4).contiguous()  # (B, C_z, H_z, W_z, C_y, k2h, k2w)
    a_flat = a_perm.reshape(-1, c_y, k2h, k2w)

    # F.conv_transpose2d with stride=1, padding=0 produces output size k2 + k1 - 1.
    # Weight shape (C_in=C_y, C_out=C_x, k1h, k1w). The math expands the kernel
    # offsets additively — exactly what we need.
    k_combined_h = k1h + k2h - 1
    k_combined_w = k1w + k2w - 1
    patch_flat = F.conv_transpose2d(a_flat, w1, stride=1, padding=0)
    # Expect output shape (-1, C_x, k_combined_h, k_combined_w).
    if patch_flat.shape[-3:] != (c_x, k_combined_h, k_combined_w):
        raise RuntimeError(
            f"Unexpected composed-patch shape {tuple(patch_flat.shape)}; expected trailing "
            f"({c_x}, {k_combined_h}, {k_combined_w})"
        )

    patches = patch_flat.reshape(B, c_z, h_z, w_z, c_x, k_combined_h, k_combined_w)
    patches = patches.reshape(*batch_shape, c_z, h_z, w_z, c_x, k_combined_h, k_combined_w)

    return Conv2dPatchOperator(
        patches=patches,
        stride=(1, 1),
        padding=(p1h + p2h, p1w + p2w),
        dilation=(1, 1),
        groups=1,
        input_shape=tuple(scaled_op.input_shape),
        output_shape=tuple(output_shape),
    )


def _compose_conv_with_patch(
    patch_op: Conv2dPatchOperator,
    weight3: torch.Tensor,
    stride3: tuple[int, int],
    padding3: tuple[int, int],
    dilation3: tuple[int, int],
    groups3: int,
    output_shape: tuple[int, ...],
) -> Conv2dPatchOperator:
    """Compose a 2D conv after a :class:`Conv2dPatchOperator` into a larger
    ``Conv2dPatchOperator`` without materializing.

    Supports stride=1, dilation=1, groups=1 on both the incoming patch op and
    the new conv₃. Effective hyperparameters of the composed patch op:

    - stride = (1, 1)
    - padding = ``patch_op.padding + padding3`` (componentwise)
    - dilation = (1, 1)
    - kernel_size = ``patch_op.kernel_size + k3 - 1`` (componentwise)

    Math (all stride-1, dilation-1 assumed)::

        z[c_z, h_z, w_z] = sum_{c_y, di_y, dj_y}
            W3[c_z, c_y, di_y, dj_y] * y[c_y, h_z + di_y - p3, w_z + dj_y - p3]

    where ``y`` is itself a patch op. Substituting and grouping by the
    effective input offset ``(m_new, n_new) = (di_y + m, dj_y + n)`` yields::

        new_patch[c_z, h_z, w_z, c_x, m_new, n_new]
            = sum_{c_y, di_y, dj_y, m, n : m_new == di_y + m, n_new == dj_y + n}
                W3[c_z, c_y, di_y, dj_y] *
                patch[c_y, h_z + di_y - p3, w_z + dj_y - p3, c_x, m, n]

    Implementation: Python loop over ``(di_y, dj_y)`` with a vectorised
    gather along ``(H_y, W_y)``, a contract against ``W3[:, :, di_y, dj_y]``,
    and a scatter-add into the new kernel slot
    ``[di_y : di_y+k_p_h, dj_y : dj_y+k_p_w]``.
    """
    if patch_op.stride != (1, 1) or stride3 != (1, 1):
        raise NotImplementedError(
            f"_compose_conv_with_patch requires stride=(1,1); got patch stride={patch_op.stride}, conv stride={stride3}"
        )
    if patch_op.dilation != (1, 1) or dilation3 != (1, 1):
        raise NotImplementedError(
            f"_compose_conv_with_patch requires dilation=(1,1); got patch dilation={patch_op.dilation}, "
            f"conv dilation={dilation3}"
        )
    if patch_op.groups != 1 or groups3 != 1:
        raise NotImplementedError("_compose_conv_with_patch requires groups=1 on both operators")
    if weight3.ndim != 4:
        raise ValueError(f"weight3 must be 4D, got {tuple(weight3.shape)}")

    p3h, p3w = padding3
    pph, ppw = patch_op.padding
    _, c_y_w3, k3h, k3w = weight3.shape
    kph, kpw = patch_op.kernel_size

    c_y_patch = int(patch_op.output_shape[-3])
    if c_y_w3 != c_y_patch:
        raise ValueError(f"Intermediate channel mismatch: weight3 expects C_y={c_y_w3}, patch_op has C_y={c_y_patch}")

    new_kh = k3h + kph - 1
    new_kw = k3w + kpw - 1
    new_padding = (p3h + pph, p3w + ppw)

    c_x = int(patch_op.input_shape[0])
    h_y = int(patch_op.output_shape[-2])
    w_y = int(patch_op.output_shape[-1])
    h_z = int(output_shape[-2])
    w_z = int(output_shape[-1])

    if tuple(output_shape[:-3]) != tuple(patch_op.output_shape[:-3]):
        raise NotImplementedError(
            "_compose_conv_with_patch does not currently support differing batch prefixes "
            f"({tuple(output_shape[:-3])} vs {tuple(patch_op.output_shape[:-3])})"
        )

    patches_old = patch_op.patches  # (*batch, C_y, H_y, W_y, C_x, kph, kpw)

    new_patches = torch.zeros(*output_shape, c_x, new_kh, new_kw, dtype=patches_old.dtype, device=patches_old.device)

    hz_range = torch.arange(h_z, device=patches_old.device)
    wz_range = torch.arange(w_z, device=patches_old.device)

    for di_y in range(k3h):
        hy = hz_range + di_y - p3h
        h_mask = (hy >= 0) & (hy < h_y)
        if not bool(h_mask.any()):
            continue
        valid_hz = hz_range[h_mask]
        valid_hy = hy[h_mask]
        h_start = int(valid_hz[0].item())
        h_end = int(valid_hz[-1].item()) + 1
        for dj_y in range(k3w):
            wy = wz_range + dj_y - p3w
            w_mask = (wy >= 0) & (wy < w_y)
            if not bool(w_mask.any()):
                continue
            valid_wz = wz_range[w_mask]
            valid_wy = wy[w_mask]
            w_start = int(valid_wz[0].item())
            w_end = int(valid_wz[-1].item()) + 1

            # Patches layout: (*batch, C_y, H_y, W_y, C_x, k_h, k_w). H_y is at
            # axis -5 (counting from the right), W_y at -4.
            sub = patches_old.index_select(-5, valid_hy).index_select(-4, valid_wy)
            # shape: (*batch, C_y, |vh|, |vw|, C_x, kph, kpw)

            w3_slice = weight3[:, :, di_y, dj_y]  # (C_z, C_y)
            contrib = torch.einsum("ZC,...Chwikl->...Zhwikl", w3_slice, sub)
            # shape: (*batch, C_z, |vh|, |vw|, C_x, kph, kpw)

            existing = new_patches[..., :, h_start:h_end, w_start:w_end, :, di_y : di_y + kph, dj_y : dj_y + kpw]
            new_patches[..., :, h_start:h_end, w_start:w_end, :, di_y : di_y + kph, dj_y : dj_y + kpw] = (
                existing + contrib
            )

    return Conv2dPatchOperator(
        patches=new_patches,
        stride=(1, 1),
        padding=new_padding,
        dilation=(1, 1),
        groups=1,
        input_shape=tuple(patch_op.input_shape),
        output_shape=tuple(output_shape),
    )


__all__ = [
    "Conv2dOperator",
    "Conv2dPatchOperator",
    "DenseOperator",
    "IdentityOperator",
    "LinearOperator",
    "ScaledConv2dOperator",
    "apply_weight_to_bounds_pair",
    "cat_output",
    "stack_output",
]
