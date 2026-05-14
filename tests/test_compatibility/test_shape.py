"""Shape compatibility for shape-manipulation ops.

PyTorch's shape ops accept the following per their docs:

* :func:`torch.reshape` / :meth:`Tensor.reshape` — any input shape, any
  target shape with matching numel; ``-1`` permitted in one position.
* :meth:`Tensor.view` — same as reshape but requires contiguous memory.
* :class:`nn.Flatten` / :func:`torch.flatten` — flattens dims
  ``start_dim:end_dim``.
* :func:`torch.cat` — concatenates along ``dim`` (default 0); all other
  dim sizes must agree.
* :func:`torch.stack` — stacks along a new ``dim``; all input shapes must
  match.
* :func:`operator.getitem` — int / slice / tuple indexing.
* :meth:`Tensor.select` / :func:`torch.select` — fixed index along
  ``dim`` (removes that dim).
* :func:`torch.unsqueeze` — inserts a size-1 dim at ``dim``.
* :func:`torch.squeeze` — removes a size-1 dim (specific or all).
* :func:`torch.transpose` / :meth:`Tensor.transpose` — swaps two dims.
* :func:`torch.permute` / :meth:`Tensor.permute` — full reordering.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ._harness import ALL_METHODS, check_op_compatibility, make_region

# -- reshape ------------------------------------------------------------------


RESHAPE_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((6,), (2, 3)),
    ((2, 3), (6,)),
    ((2, 3), (3, 2)),
    ((2, 3, 4), (6, 4)),
    ((2, 3, 4), (2, 12)),
    ((2, 3, 4), (24,)),
    ((2, 3, 4), (-1, 4)),
]


def _reshape_factory(out: tuple[int, ...]):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, out)

    return fn


@pytest.mark.parametrize(("in_shape", "out_shape"), RESHAPE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_reshape(method: str, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> None:
    region = make_region(in_shape)
    check_op_compatibility(_reshape_factory(out_shape), (torch.zeros(in_shape),), [region], method)


def _view_factory(out: tuple[int, ...]):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.view(out)

    return fn


@pytest.mark.parametrize(("in_shape", "out_shape"), RESHAPE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_view(method: str, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> None:
    region = make_region(in_shape)
    check_op_compatibility(_view_factory(out_shape), (torch.zeros(in_shape),), [region], method)


# -- flatten ------------------------------------------------------------------


FLATTEN_CASES: list[tuple[tuple[int, ...], int, int]] = [
    ((6,), 0, -1),
    ((2, 3), 0, -1),
    ((2, 3), 0, 1),
    ((2, 3, 4), 0, -1),
    ((2, 3, 4), 1, 2),
    ((2, 3, 4), 0, 1),
]


def _flatten_factory(start_dim: int, end_dim: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

    return fn


@pytest.mark.parametrize(("shape", "start_dim", "end_dim"), FLATTEN_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_flatten(method: str, shape: tuple[int, ...], start_dim: int, end_dim: int) -> None:
    region = make_region(shape)
    check_op_compatibility(_flatten_factory(start_dim, end_dim), (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nn_flatten_default(method: str, shape: tuple[int, ...]) -> None:
    # nn.Flatten defaults to start_dim=1, end_dim=-1 — i.e. flatten everything
    # except the leading batch dim. Even with no explicit batch dim, the op
    # itself must accept this rank-preserving rewrite.
    region = make_region(shape)
    check_op_compatibility(nn.Flatten(), (torch.zeros(shape),), [region], method)


# -- cat ----------------------------------------------------------------------


CAT_CASES: list[tuple[tuple[int, ...], tuple[int, ...], int]] = [
    ((3,), (4,), 0),
    ((2, 3), (2, 5), 1),
    ((2, 3), (4, 3), 0),
    ((2, 3, 4), (2, 3, 5), 2),
    ((2, 3, 4), (2, 3, 5), -1),
]


def _cat_factory(dim: int):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y], dim=dim)

    return fn


@pytest.mark.parametrize(("a_shape", "b_shape", "dim"), CAT_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_cat(
    method: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dim: int,
) -> None:
    regions = [make_region(a_shape), make_region(b_shape)]
    dummies = (torch.zeros(a_shape), torch.zeros(b_shape))
    check_op_compatibility(_cat_factory(dim), dummies, regions, method)


# -- stack --------------------------------------------------------------------


STACK_CASES: list[tuple[tuple[int, ...], int]] = [
    ((3,), 0),
    ((3,), 1),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3, 4), 0),
    ((2, 3, 4), -1),
]


def _stack_factory(dim: int):
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([x, y], dim=dim)

    return fn


@pytest.mark.parametrize(("shape", "dim"), STACK_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_stack(method: str, shape: tuple[int, ...], dim: int) -> None:
    regions = [make_region(shape), make_region(shape)]
    dummies = (torch.zeros(shape), torch.zeros(shape))
    check_op_compatibility(_stack_factory(dim), dummies, regions, method)


# -- getitem ------------------------------------------------------------------


GETITEM_CASES: list[tuple[tuple[int, ...], object, tuple[int, ...]]] = [
    ((4,), 2, ()),
    ((4,), slice(1, 3), (2,)),
    ((4,), slice(None), (4,)),
    ((2, 3), 0, (3,)),
    ((2, 3), (0, 1), ()),
    ((2, 3), (slice(None), 1), (2,)),
    ((2, 3), (slice(0, 1), slice(None)), (1, 3)),
    ((2, 3, 4), (0, slice(None), 2), (3,)),
]


def _getitem_factory(index: object):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x[index]

    return fn


@pytest.mark.parametrize(("shape", "index", "expected_out"), GETITEM_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_getitem(
    method: str,
    shape: tuple[int, ...],
    index: object,
    expected_out: tuple[int, ...],
) -> None:
    del expected_out  # The harness asserts the eager-output shape itself.
    region = make_region(shape)
    check_op_compatibility(_getitem_factory(index), (torch.zeros(shape),), [region], method)


# -- select -------------------------------------------------------------------


SELECT_CASES: list[tuple[tuple[int, ...], int, int]] = [
    ((4,), 0, 1),
    ((2, 3), 0, 1),
    ((2, 3), 1, 2),
    ((2, 3), -1, 0),
    ((2, 3, 4), 0, 0),
    ((2, 3, 4), 2, 3),
]


def _select_factory(dim: int, index: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.select(dim, index)

    return fn


@pytest.mark.parametrize(("shape", "dim", "index"), SELECT_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_select_method(method: str, shape: tuple[int, ...], dim: int, index: int) -> None:
    region = make_region(shape)
    check_op_compatibility(_select_factory(dim, index), (torch.zeros(shape),), [region], method)


# -- unsqueeze / squeeze ------------------------------------------------------


UNSQUEEZE_CASES: list[tuple[tuple[int, ...], int]] = [
    ((), 0),
    ((3,), 0),
    ((3,), 1),
    ((3,), -1),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3), -1),
]


def _unsqueeze_factory(dim: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(x, dim)

    return fn


@pytest.mark.parametrize(("shape", "dim"), UNSQUEEZE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_unsqueeze(method: str, shape: tuple[int, ...], dim: int) -> None:
    region = make_region(shape)
    check_op_compatibility(_unsqueeze_factory(dim), (torch.zeros(shape),), [region], method)


SQUEEZE_CASES: list[tuple[tuple[int, ...], int | None]] = [
    ((1,), 0),
    ((1, 3), 0),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((2, 1, 3, 1), None),
    ((1, 1, 3), None),
]


def _squeeze_factory(dim: int | None):
    if dim is None:

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(x)

        return fn

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, dim)

    return fn


@pytest.mark.parametrize(("shape", "dim"), SQUEEZE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_squeeze(method: str, shape: tuple[int, ...], dim: int | None) -> None:
    region = make_region(shape)
    check_op_compatibility(_squeeze_factory(dim), (torch.zeros(shape),), [region], method)


# -- transpose ----------------------------------------------------------------


TRANSPOSE_CASES: list[tuple[tuple[int, ...], int, int]] = [
    ((2, 3), 0, 1),
    ((2, 3), 1, 0),
    ((2, 3, 4), 0, 1),
    ((2, 3, 4), 0, 2),
    ((2, 3, 4), 1, 2),
    ((2, 3, 4), -1, -2),
]


def _transpose_method_factory(dim0: int, dim1: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(dim0, dim1)

    return fn


def _transpose_function_factory(dim0: int, dim1: int):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim0, dim1)

    return fn


@pytest.mark.parametrize(("shape", "dim0", "dim1"), TRANSPOSE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_transpose_method(method: str, shape: tuple[int, ...], dim0: int, dim1: int) -> None:
    region = make_region(shape)
    check_op_compatibility(
        _transpose_method_factory(dim0, dim1),
        (torch.zeros(shape),),
        [region],
        method,
    )


@pytest.mark.parametrize(("shape", "dim0", "dim1"), TRANSPOSE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_transpose_function(method: str, shape: tuple[int, ...], dim0: int, dim1: int) -> None:
    region = make_region(shape)
    check_op_compatibility(
        _transpose_function_factory(dim0, dim1),
        (torch.zeros(shape),),
        [region],
        method,
    )


# -- permute ------------------------------------------------------------------


PERMUTE_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((2, 3), (1, 0)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (0, 2, 1)),
    ((2, 3, 4), (-1, -2, -3)),
    ((2, 3, 4, 5), (3, 1, 2, 0)),
]


def _permute_method_factory(dims: tuple[int, ...]):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.permute(*dims)

    return fn


def _permute_function_factory(dims: tuple[int, ...]):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, dims)

    return fn


@pytest.mark.parametrize(("shape", "dims"), PERMUTE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_permute_method(method: str, shape: tuple[int, ...], dims: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_permute_method_factory(dims), (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize(("shape", "dims"), PERMUTE_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_permute_function(method: str, shape: tuple[int, ...], dims: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_permute_function_factory(dims), (torch.zeros(shape),), [region], method)
