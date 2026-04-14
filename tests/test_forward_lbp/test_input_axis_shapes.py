from __future__ import annotations

import pytest
import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp import (
    ForwardLBPAbs,
    ForwardLBPAdd,
    ForwardLBPClamp,
    ForwardLBPConcat,
    ForwardLBPCos,
    ForwardLBPDiv,
    ForwardLBPExp,
    ForwardLBPFlatten,
    ForwardLBPGetItem,
    ForwardLBPLinear,
    ForwardLBPLog,
    ForwardLBPMatmul,
    ForwardLBPMax,
    ForwardLBPMaximum,
    ForwardLBPMean,
    ForwardLBPMin,
    ForwardLBPMinimum,
    ForwardLBPMul,
    ForwardLBPNeg,
    ForwardLBPPermute,
    ForwardLBPReciprocal,
    ForwardLBPRelu,
    ForwardLBPReshape,
    ForwardLBPSelect,
    ForwardLBPSigmoid,
    ForwardLBPSin,
    ForwardLBPSqrt,
    ForwardLBPSqueeze,
    ForwardLBPStack,
    ForwardLBPSub,
    ForwardLBPSum,
    ForwardLBPTan,
    ForwardLBPTanh,
    ForwardLBPTranspose,
    ForwardLBPUnsqueeze,
    ForwardLBPView,
)
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate

INPUT_SHAPES: list[tuple[int, ...]] = [(), (3,), (2, 2)]
OPERATIONS: list[str] = [
    "abs",
    "add",
    "clamp",
    "cat",
    "cos",
    "div",
    "exp",
    "flatten",
    "getitem",
    "linear",
    "log",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mul",
    "neg",
    "permute",
    "reciprocal",
    "relu",
    "reshape",
    "select",
    "sigmoid",
    "sin",
    "sqrt",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "tan",
    "tanh",
    "transpose",
    "unsqueeze",
    "view",
]


def _tensor_full(shape: tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value) if shape else torch.tensor(value)


def make_bounds(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    input_id: int = 1,
    lower_value: float = 0.1,
    upper_value: float = 1.1,
) -> LinearBounds:
    region = HyperRectangle(
        lower=_tensor_full(input_shape, lower_value),
        upper=_tensor_full(input_shape, upper_value),
    )

    coeff_shape = output_shape + input_shape
    linear_lower = [torch.full(coeff_shape if coeff_shape else (), 0.1)]
    linear_upper = [torch.full(coeff_shape if coeff_shape else (), 0.2)]
    bias_lower = torch.zeros(output_shape)
    bias_upper = torch.full(output_shape, 0.5)

    return LinearBounds(
        regions=[region],
        linear_lower=linear_lower,
        bias_lower=bias_lower,
        linear_upper=linear_upper,
        bias_upper=bias_upper,
        input_ids=[input_id],
    )


def make_peer(bounds: LinearBounds, *, input_id: int) -> LinearBounds:
    input_shape = tuple(bounds.regions[0].shape)
    output_shape = tuple(bounds.bias_lower.shape)
    return make_bounds(input_shape, output_shape, input_id=input_id)


def make_identity_bounds(input_shape: tuple[int, ...], *, input_id: int = 1) -> LinearBounds:
    region = HyperRectangle(
        lower=_tensor_full(input_shape, 0.1),
        upper=_tensor_full(input_shape, 1.1),
    )

    if input_shape:
        in_features = torch.Size(input_shape).numel()
        identity = torch.eye(in_features).reshape(*input_shape, *input_shape)
    else:
        identity = torch.tensor(1.0)

    bias = torch.zeros(input_shape)

    return LinearBounds(
        regions=[region],
        linear_lower=[identity],
        bias_lower=bias,
        linear_upper=[identity],
        bias_upper=bias,
        input_ids=[input_id],
    )


def assert_input_axes_match_regions(bounds: LinearBounds) -> None:
    for region, lower, upper in zip(bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True):
        expected = torch.Size(region.shape)
        lower_input_axes = torch.Size(lower.shape[len(bounds.bias_lower.shape) :])
        upper_input_axes = torch.Size(upper.shape[len(bounds.bias_upper.shape) :])
        assert lower_input_axes == expected
        assert upper_input_axes == expected


def run_operation(operation: str, base: LinearBounds) -> LinearBounds:
    nonlinear_base = make_identity_bounds(tuple(base.regions[0].shape), input_id=1)

    if operation == "abs":
        return propagate(ForwardLBPAbs(), nonlinear_base)
    if operation == "add":
        return propagate(ForwardLBPAdd(), base, make_peer(base, input_id=2))
    if operation == "clamp":
        return propagate(ForwardLBPClamp(), nonlinear_base, min=0.0, max=1.0)
    if operation == "cat":
        peer = make_peer(base, input_id=2)
        return propagate(ForwardLBPConcat(), [base, peer], dim=0)
    if operation == "cos":
        return propagate(ForwardLBPCos(), nonlinear_base)
    if operation == "div":
        return propagate(ForwardLBPDiv(), base, 2.0)
    if operation == "exp":
        return propagate(ForwardLBPExp(), nonlinear_base)
    if operation == "flatten":
        return propagate(ForwardLBPFlatten(), base, start_dim=0, end_dim=-1)
    if operation == "getitem":
        return propagate(ForwardLBPGetItem(), base, (1, 1))
    if operation == "linear":
        b = make_bounds(tuple(base.regions[0].shape), (2, 4), input_id=1)
        return propagate(ForwardLBPLinear(), b, torch.ones(5, 4), torch.zeros(5))
    if operation == "log":
        return propagate(ForwardLBPLog(), nonlinear_base)
    if operation == "matmul":
        b = make_bounds(tuple(base.regions[0].shape), (2, 4), input_id=1)
        return propagate(ForwardLBPMatmul(), b, torch.ones(4, 5))
    if operation == "max":
        return propagate(ForwardLBPMax(), base)
    if operation == "maximum":
        return propagate(ForwardLBPMaximum(), base, make_peer(base, input_id=2))
    if operation == "mean":
        return propagate(ForwardLBPMean(), base)
    if operation == "min":
        return propagate(ForwardLBPMin(), base)
    if operation == "minimum":
        return propagate(ForwardLBPMinimum(), base, make_peer(base, input_id=2))
    if operation == "mul":
        return propagate(ForwardLBPMul(), base, 2.0)
    if operation == "neg":
        return propagate(ForwardLBPNeg(), base)
    if operation == "permute":
        b = make_bounds(tuple(base.regions[0].shape), (2, 3), input_id=1)
        return propagate(ForwardLBPPermute(), b, 1, 0)
    if operation == "reciprocal":
        return propagate(ForwardLBPReciprocal(), nonlinear_base)
    if operation == "relu":
        return propagate(ForwardLBPRelu(), nonlinear_base)
    if operation == "reshape":
        b = make_bounds(tuple(base.regions[0].shape), (2, 3), input_id=1)
        return propagate(ForwardLBPReshape(), b, (3, 2))
    if operation == "select":
        b = make_bounds(tuple(base.regions[0].shape), (1,), input_id=1)
        return propagate(ForwardLBPSelect(), b, 0, 0)
    if operation == "sigmoid":
        return propagate(ForwardLBPSigmoid(), nonlinear_base)
    if operation == "sin":
        return propagate(ForwardLBPSin(), nonlinear_base)
    if operation == "sqrt":
        return propagate(ForwardLBPSqrt(), nonlinear_base)
    if operation == "squeeze":
        b = make_bounds(tuple(base.regions[0].shape), (1,), input_id=1)
        return propagate(ForwardLBPSqueeze(), b, dim=0)
    if operation == "stack":
        peer = make_peer(base, input_id=2)
        return propagate(ForwardLBPStack(), [base, peer], dim=0)
    if operation == "sub":
        return propagate(ForwardLBPSub(), base, make_peer(base, input_id=2))
    if operation == "sum":
        return propagate(ForwardLBPSum(), base)
    if operation == "tan":
        return propagate(ForwardLBPTan(), nonlinear_base)
    if operation == "tanh":
        return propagate(ForwardLBPTanh(), nonlinear_base)
    if operation == "transpose":
        b = make_bounds(tuple(base.regions[0].shape), (2, 3), input_id=1)
        return propagate(ForwardLBPTranspose(), b, 0, 1)
    if operation == "unsqueeze":
        return propagate(ForwardLBPUnsqueeze(), base, dim=1)
    if operation == "view":
        b = make_bounds(tuple(base.regions[0].shape), (2, 3), input_id=1)
        return propagate(ForwardLBPView(), b, (3, 2))

    raise ValueError(f"Unhandled operation: {operation}")


def expected_structure(operation: str, input_shape: tuple[int, ...]) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    if operation in {
        "abs",
        "clamp",
        "cos",
        "exp",
        "log",
        "reciprocal",
        "relu",
        "sigmoid",
        "sin",
        "sqrt",
        "tan",
        "tanh",
    }:
        return input_shape, [input_shape]

    if operation in {"add", "sub"}:
        return (2, 3), [input_shape, input_shape]

    if operation == "cat":
        return (4, 3), [input_shape, input_shape]

    if operation == "stack":
        return (2, 2, 3), []

    if operation in {"maximum", "minimum"}:
        return (2, 3), []

    if operation in {"max", "mean", "min", "sum"}:
        return (), []

    if operation in {"div", "mul", "neg"}:
        return (2, 3), [input_shape]

    if operation == "flatten":
        return (6,), [input_shape]

    if operation in {"getitem", "select", "squeeze"}:
        return (), [input_shape]

    if operation in {"linear", "matmul"}:
        return (2, 5), [input_shape]

    if operation in {"permute", "reshape", "transpose", "view"}:
        return (3, 2), [input_shape]

    if operation == "unsqueeze":
        return (2, 1, 3), [input_shape]

    raise ValueError(f"No expected structure defined for operation {operation!r}")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_forward_lbp_operations_support_varied_input_axes(
    input_shape: tuple[int, ...],
    operation: str,
) -> None:
    base = make_bounds(input_shape=input_shape, output_shape=(2, 3))
    result = run_operation(operation, base)
    expected_bias_shape, expected_region_shapes = expected_structure(operation, input_shape)

    assert isinstance(result, LinearBounds)
    assert tuple(result.bias_lower.shape) == expected_bias_shape
    assert tuple(result.bias_upper.shape) == expected_bias_shape
    assert [tuple(region.shape) for region in result.regions] == expected_region_shapes
    assert len(result.input_ids) == len(expected_region_shapes)
    assert len(set(result.input_ids)) == len(result.input_ids)

    if expected_region_shapes:
        assert len(result.linear_lowers) == len(expected_region_shapes)
        assert len(result.linear_uppers) == len(expected_region_shapes)
        for region_shape, lower, upper in zip(
            expected_region_shapes,
            result.linear_lowers,
            result.linear_uppers,
            strict=True,
        ):
            assert tuple(lower.shape) == expected_bias_shape + region_shape
            assert tuple(upper.shape) == expected_bias_shape + region_shape
    else:
        assert result.linear_lowers == []
        assert result.linear_uppers == []

    lower, upper = result.concretize()
    assert tuple(lower.shape) == expected_bias_shape
    assert tuple(upper.shape) == expected_bias_shape
    assert torch.all(lower <= upper + 1e-6)
