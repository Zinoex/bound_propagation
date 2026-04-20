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
        return self._inner.apply(x).reshape(self._output_shape)

    def apply_transpose(self, y: torch.Tensor) -> torch.Tensor:
        out_ndim = len(self._output_shape)
        if y.ndim < out_ndim or tuple(y.shape[y.ndim - out_ndim :]) != tuple(self._output_shape):
            raise ValueError(
                f"ReshapeOperator.apply_transpose: y.shape trailing "
                f"{tuple(y.shape[y.ndim - out_ndim :])} does not match output_shape "
                f"{tuple(self._output_shape)}"
            )
        leading = y.shape[: y.ndim - out_ndim]
        reshaped = y.reshape(*leading, *self._inner.output_shape)
        return self._inner.apply_transpose(reshaped)

    def concretize_min(self, region: SimpleRegion) -> torch.Tensor:
        return self._inner.concretize_min(region).reshape(self._output_shape)

    def concretize_max(self, region: SimpleRegion) -> torch.Tensor:
        return self._inner.concretize_max(region).reshape(self._output_shape)

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
        if (
            isinstance(other, ReshapeOperator)
            and other._output_shape == self._output_shape
            and other._inner.output_shape == self._inner.output_shape
            and other._inner.input_shape == self._inner.input_shape
        ):
            combined_inner = self._inner.add(other._inner)
            return ReshapeOperator(combined_inner, self._output_shape)
        return self.to_dense().add(other.to_dense())

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
