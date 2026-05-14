"""Shape compatibility for linear / matmul ops.

PyTorch documents:

* :class:`torch.nn.Linear` (and :func:`F.linear`) — input shape
  ``(*, H_in)``; output shape ``(*, H_out)``. Any number of leading dims
  (including zero) is allowed.
* :func:`torch.matmul` — supports five distinct cases by combined rank:

  * 1-D vs 1-D: dot product, scalar output.
  * 2-D vs 2-D: matrix product.
  * 1-D vs 2-D: 1-D is prepended with a dim of 1 then removed.
  * 2-D vs 1-D: matrix-vector.
  * ND vs ND (with N >= 1, M >= 1): batched matmul with broadcasting of
    the leading dims.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ._harness import ALL_METHODS, check_op_compatibility, make_region

# -- nn.Linear / F.linear -----------------------------------------------------

# Feature shape that ends with H_in=3; the leading dims can be anything.
LINEAR_INPUT_SHAPES: list[tuple[int, ...]] = [
    (3,),
    (2, 3),
    (4, 2, 3),
    (2, 4, 2, 3),
]


@pytest.mark.parametrize("shape", LINEAR_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nn_linear(method: str, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 5)
    region = make_region(shape, lower=-0.5, upper=0.5)
    check_op_compatibility(model, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", LINEAR_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nn_linear_no_bias(method: str, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 5, bias=False)
    region = make_region(shape, lower=-0.5, upper=0.5)
    check_op_compatibility(model, (torch.zeros(shape),), [region], method)


# -- torch.matmul -------------------------------------------------------------
#
# The five matmul shape cases that PyTorch documents.

MATMUL_CASES: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
    ("dot_1d_1d", (4,), (4,)),
    ("matvec_2d_1d", (3, 4), (4,)),
    ("vecmat_1d_2d", (4,), (4, 5)),
    ("mat_2d_2d", (3, 4), (4, 5)),
    ("batched_3d_3d", (2, 3, 4), (2, 4, 5)),
    ("batched_broadcast_3d_3d", (2, 3, 4), (1, 4, 5)),
    ("batched_broadcast_3d_2d", (2, 3, 4), (4, 5)),
    ("batched_4d_4d", (2, 3, 4, 5), (2, 3, 5, 6)),
]


def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y)


@pytest.mark.parametrize(("name", "a_shape", "b_shape"), MATMUL_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_matmul_two_abstract(
    method: str,
    name: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
) -> None:
    del name
    regions = [make_region(a_shape, lower=-0.5, upper=0.5), make_region(b_shape, lower=-0.5, upper=0.5)]
    dummies = (torch.zeros(a_shape), torch.zeros(b_shape))
    check_op_compatibility(_matmul, dummies, regions, method)


def _matmul_const_right(x: torch.Tensor) -> torch.Tensor:
    weight = torch.full((4, 5), 0.5)
    return torch.matmul(x, weight)


def _matmul_const_left(x: torch.Tensor) -> torch.Tensor:
    weight = torch.full((5, 4), 0.5)
    return torch.matmul(weight, x)


MATMUL_CONST_RIGHT_INPUT_SHAPES: list[tuple[int, ...]] = [
    (4,),
    (3, 4),
    (2, 3, 4),
    (2, 1, 3, 4),
]


@pytest.mark.parametrize("shape", MATMUL_CONST_RIGHT_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_matmul_constant_right(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-0.5, upper=0.5)
    check_op_compatibility(_matmul_const_right, (torch.zeros(shape),), [region], method)


MATMUL_CONST_LEFT_INPUT_SHAPES: list[tuple[int, ...]] = [
    (4,),
    (4, 3),
]


@pytest.mark.parametrize("shape", MATMUL_CONST_LEFT_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_matmul_constant_left(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-0.5, upper=0.5)
    check_op_compatibility(_matmul_const_left, (torch.zeros(shape),), [region], method)
