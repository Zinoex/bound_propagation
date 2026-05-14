"""Shape compatibility for reduction ops.

PyTorch's reductions (``sum``, ``mean``, ``amax``, ``amin``) accept any
shape. They optionally take:

* a ``dim`` int or tuple-of-ints, reducing over those dims;
* a ``keepdim`` bool that preserves the reduced dims as size 1;
* full reduction when ``dim`` is omitted.

Each combination should produce a tensor matching eager PyTorch.
"""

from __future__ import annotations

import pytest
import torch

from ._harness import ALL_METHODS, check_op_compatibility, make_region

REDUCTION_INPUT_SHAPES: list[tuple[int, ...]] = [
    (4,),
    (2, 3),
    (2, 3, 4),
]


def _sum_full(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x)


def _mean_full(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x)


def _amax_full(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(x)


def _amin_full(x: torch.Tensor) -> torch.Tensor:
    return torch.amin(x)


@pytest.mark.parametrize("shape", REDUCTION_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sum_full_reduction(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_sum_full, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", REDUCTION_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_mean_full_reduction(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_mean_full, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", REDUCTION_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_amax_full_reduction(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_amax_full, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", REDUCTION_INPUT_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_amin_full_reduction(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_amin_full, (torch.zeros(shape),), [region], method)


# -- Per-dim reductions -------------------------------------------------------

DIM_CASES: list[tuple[tuple[int, ...], int, bool]] = [
    ((4,), 0, False),
    ((4,), 0, True),
    ((4,), -1, False),
    ((2, 3), 0, False),
    ((2, 3), 1, False),
    ((2, 3), -1, True),
    ((2, 3, 4), 0, False),
    ((2, 3, 4), 1, False),
    ((2, 3, 4), 2, True),
    ((2, 3, 4), -1, False),
]


def _factory_sum(dim: int, keepdim: bool):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=dim, keepdim=keepdim)

    return fn


def _factory_mean(dim: int, keepdim: bool):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=dim, keepdim=keepdim)

    return fn


def _factory_amax(dim: int, keepdim: bool):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=dim, keepdim=keepdim)

    return fn


def _factory_amin(dim: int, keepdim: bool):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=dim, keepdim=keepdim)

    return fn


@pytest.mark.parametrize(("shape", "dim", "keepdim"), DIM_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sum_dim_reduction(method: str, shape: tuple[int, ...], dim: int, keepdim: bool) -> None:
    region = make_region(shape)
    check_op_compatibility(_factory_sum(dim, keepdim), (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize(("shape", "dim", "keepdim"), DIM_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_mean_dim_reduction(method: str, shape: tuple[int, ...], dim: int, keepdim: bool) -> None:
    region = make_region(shape)
    check_op_compatibility(_factory_mean(dim, keepdim), (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize(("shape", "dim", "keepdim"), DIM_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_amax_dim_reduction(method: str, shape: tuple[int, ...], dim: int, keepdim: bool) -> None:
    region = make_region(shape)
    check_op_compatibility(_factory_amax(dim, keepdim), (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize(("shape", "dim", "keepdim"), DIM_CASES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_amin_dim_reduction(method: str, shape: tuple[int, ...], dim: int, keepdim: bool) -> None:
    region = make_region(shape)
    check_op_compatibility(_factory_amin(dim, keepdim), (torch.zeros(shape),), [region], method)
