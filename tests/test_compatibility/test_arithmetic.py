"""Shape compatibility for binary arithmetic ops.

PyTorch's binary arithmetic ops (``torch.add``, ``sub``, ``mul``, ``div``,
``maximum``, ``minimum``, ``pow``) support full broadcasting: any two
shapes that broadcast against each other are accepted, including the
degenerate (scalar, anything) case.

Tests cover:

* Same-shape pairs, including scalars.
* Broadcasting pairs: ``(N,)`` vs ``(M, N)``, ``(1, N)`` vs ``(M, N)``,
  ``(M, 1)`` vs ``(M, N)``, etc.
* Scalar-vs-tensor (constant + abstract) — already a separate code path
  in most strategies.
"""

from __future__ import annotations

import pytest
import torch

from ._harness import ALL_METHODS, check_op_compatibility, make_region, safe_positive_region

# Pairs of (x_shape, y_shape) where each pair must broadcast under PyTorch
# semantics. Includes same-shape, scalar/tensor, and broadcasting cases.
BINARY_SHAPE_PAIRS: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((), ()),
    ((3,), (3,)),
    ((2, 3), (2, 3)),
    ((2, 3, 4), (2, 3, 4)),
    ((3,), (2, 3)),
    ((1, 3), (2, 3)),
    ((2, 1), (2, 3)),
    ((2, 1, 4), (2, 3, 4)),
    ((1,), (2, 3)),
]


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def _sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def _mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def _div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / y


def _maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


def _minimum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_add(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    regions = [make_region(a), make_region(b)]
    dummies = (torch.zeros(a), torch.zeros(b))
    check_op_compatibility(_add, dummies, regions, method)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sub(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    regions = [make_region(a), make_region(b)]
    dummies = (torch.zeros(a), torch.zeros(b))
    check_op_compatibility(_sub, dummies, regions, method)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_mul(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    regions = [make_region(a), make_region(b)]
    dummies = (torch.zeros(a), torch.zeros(b))
    check_op_compatibility(_mul, dummies, regions, method)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_div(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    # Denominator must stay away from zero across the region.
    regions = [make_region(a), safe_positive_region(b)]
    dummies = (torch.zeros(a), torch.ones(b))
    check_op_compatibility(_div, dummies, regions, method)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_maximum(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    regions = [make_region(a), make_region(b)]
    dummies = (torch.zeros(a), torch.zeros(b))
    check_op_compatibility(_maximum, dummies, regions, method)


@pytest.mark.parametrize("pair", BINARY_SHAPE_PAIRS)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_minimum(method: str, pair: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    a, b = pair
    regions = [make_region(a), make_region(b)]
    dummies = (torch.zeros(a), torch.zeros(b))
    check_op_compatibility(_minimum, dummies, regions, method)


# -- Constant + abstract variants ---------------------------------------------
#
# Many strategies branch on whether the second operand is a literal scalar /
# tensor. The single-abstract-input tests below cover those code paths.


CONSTANT_OP_SHAPES: list[tuple[int, ...]] = [
    (),
    (4,),
    (2, 3),
    (2, 3, 4),
]


def _add_const(x: torch.Tensor) -> torch.Tensor:
    return x + 1.5


def _sub_const(x: torch.Tensor) -> torch.Tensor:
    return x - 1.5


def _const_sub(x: torch.Tensor) -> torch.Tensor:
    return 1.5 - x


def _mul_const(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


def _const_div(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / x


@pytest.mark.parametrize("shape", CONSTANT_OP_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_add_constant(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_add_const, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", CONSTANT_OP_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sub_constant(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_sub_const, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", CONSTANT_OP_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_constant_minus_abstract(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_const_sub, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", CONSTANT_OP_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_mul_constant(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_mul_const, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", CONSTANT_OP_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_constant_div_abstract(method: str, shape: tuple[int, ...]) -> None:
    region = safe_positive_region(shape)
    check_op_compatibility(_const_div, (torch.ones(shape),), [region], method)
