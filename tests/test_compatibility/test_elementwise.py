"""Shape compatibility for element-wise ops.

PyTorch documents every element-wise op (relu, sigmoid, tanh, exp, log,
sqrt, reciprocal, abs, clamp, sin, cos, tan, pow, neg) as accepting any
tensor shape, including scalars (0-D) and higher-rank tensors.

Each op gets parametrized over a set of representative shapes — scalar,
1-D, 2-D, 3-D, 4-D — and every :class:`BoundPropagator` subclass is
exercised on each.
"""

from __future__ import annotations

import pytest
import torch

from ._harness import ALL_METHODS, check_op_compatibility, make_region, safe_positive_region

# Element-wise ops accept arbitrary shapes; pick a representative set
# spanning rank 0 (scalar) to rank 4.
ELEMENTWISE_SHAPES: list[tuple[int, ...]] = [
    (),
    (4,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 2),
]


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _neg(x: torch.Tensor) -> torch.Tensor:
    return torch.neg(x)


def _abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def _exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def _sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def _cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)


def _tan(x: torch.Tensor) -> torch.Tensor:
    return torch.tan(x)


def _log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def _sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


def _reciprocal(x: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(x)


def _clamp(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=-0.5, max=0.5)


def _pow_constant(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x, 2)


# -- Unary, defined on all of R -----------------------------------------------


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_relu(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_relu, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sigmoid(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_sigmoid, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_tanh(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_tanh, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_neg(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_neg, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_abs(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape)
    check_op_compatibility(_abs, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_exp(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-1.0, upper=1.0)
    check_op_compatibility(_exp, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sin(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-1.0, upper=1.0)
    check_op_compatibility(_sin, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_cos(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-1.0, upper=1.0)
    check_op_compatibility(_cos, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_tan(method: str, shape: tuple[int, ...]) -> None:
    # tan has poles at +/-pi/2; restrict region well inside.
    region = make_region(shape, lower=-1.0, upper=1.0)
    check_op_compatibility(_tan, (torch.zeros(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_clamp(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-2.0, upper=2.0)
    check_op_compatibility(_clamp, (torch.zeros(shape),), [region], method)


# -- Unary, requiring positive domain -----------------------------------------


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_log(method: str, shape: tuple[int, ...]) -> None:
    region = safe_positive_region(shape)
    check_op_compatibility(_log, (torch.ones(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_sqrt(method: str, shape: tuple[int, ...]) -> None:
    region = safe_positive_region(shape)
    check_op_compatibility(_sqrt, (torch.ones(shape),), [region], method)


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_reciprocal(method: str, shape: tuple[int, ...]) -> None:
    region = safe_positive_region(shape)
    check_op_compatibility(_reciprocal, (torch.ones(shape),), [region], method)


# -- pow: constant exponent ---------------------------------------------------


@pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
@pytest.mark.parametrize("method", ALL_METHODS)
def test_pow_constant_exponent(method: str, shape: tuple[int, ...]) -> None:
    region = make_region(shape, lower=-1.0, upper=1.0)
    check_op_compatibility(_pow_constant, (torch.zeros(shape),), [region], method)
