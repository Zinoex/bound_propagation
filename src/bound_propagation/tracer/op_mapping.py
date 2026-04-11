"""
Mapping from PyTorch operations to IR OperationType.

This module provides the mapping between torch operations (from torch.fx)
and our internal OperationType enum.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ..ir import OperationType

# Mapping from torch function to OperationType
TORCH_OP_MAPPING: dict[Callable[..., Any] | str, OperationType] = {
    # Linear operations
    torch.matmul: OperationType.MATMUL,
    torch.mm: OperationType.MATMUL,
    torch.bmm: OperationType.MATMUL,
    operator.matmul: OperationType.MATMUL,
    F.linear: OperationType.LINEAR,
    torch.transpose: OperationType.TRANSPOSE,
    torch.permute: OperationType.PERMUTE,
    # Elementwise arithmetic
    torch.add: OperationType.ADD,
    operator.add: OperationType.ADD,
    torch.sub: OperationType.SUB,
    operator.sub: OperationType.SUB,
    torch.mul: OperationType.MUL,
    operator.mul: OperationType.MUL,
    torch.div: OperationType.DIV,
    operator.truediv: OperationType.DIV,
    torch.pow: OperationType.POW,
    operator.pow: OperationType.POW,
    torch.neg: OperationType.NEG,
    operator.neg: OperationType.NEG,
    torch.reciprocal: OperationType.RECIPROCAL,
    # Activations
    torch.relu: OperationType.RELU,
    F.relu: OperationType.RELU,
    torch.sigmoid: OperationType.SIGMOID,
    F.sigmoid: OperationType.SIGMOID,
    torch.tanh: OperationType.TANH,
    F.tanh: OperationType.TANH,
    torch.exp: OperationType.EXP,
    torch.log: OperationType.LOG,
    torch.sqrt: OperationType.SQRT,
    torch.abs: OperationType.ABS,
    torch.clamp: OperationType.CLAMP,
    torch.heaviside: OperationType.CLAMP,
    # Trigonometric
    torch.sin: OperationType.SIN,
    torch.cos: OperationType.COS,
    torch.tan: OperationType.TAN,
    # Reductions
    torch.sum: OperationType.SUM,
    torch.mean: OperationType.MEAN,
    torch.max: OperationType.MAX,
    torch.min: OperationType.MIN,
    # Structural
    torch.cat: OperationType.CONCAT,
    torch.concat: OperationType.CONCAT,
    torch.split: OperationType.SPLIT,
    torch.gather: OperationType.GATHER,
    torch.reshape: OperationType.RESHAPE,
    torch.flatten: OperationType.FLATTEN,
    torch.unsqueeze: OperationType.UNSQUEEZE,
    torch.squeeze: OperationType.SQUEEZE,
    # Indexing/selection
    operator.getitem: OperationType.SELECT,
}

# Mapping from torch.nn.Module types to OperationType
MODULE_OP_MAPPING: dict[type, OperationType] = {
    torch.nn.Linear: OperationType.LINEAR,
    torch.nn.ReLU: OperationType.RELU,
    torch.nn.Sigmoid: OperationType.SIGMOID,
    torch.nn.Tanh: OperationType.TANH,
    torch.nn.Flatten: OperationType.FLATTEN,
}


def get_operation_type(target: Any) -> OperationType | None:
    """
    Get OperationType for a torch.fx node target.

    Args:
        target: The target from a torch.fx.Node (function, method, or module)

    Returns:
        Corresponding OperationType, or None if not recognized

    Examples:
        >>> get_operation_type(torch.relu)
        <OperationType.RELU: 'relu'>
        >>> get_operation_type(torch.nn.Linear)
        <OperationType.LINEAR: 'linear'>
        >>> get_operation_type("unknown_op")
        None
    """
    # Direct lookup in function mapping
    if target in TORCH_OP_MAPPING:
        return TORCH_OP_MAPPING[target]

    # Check if it's a module class
    if isinstance(target, type) and issubclass(target, torch.nn.Module):
        return MODULE_OP_MAPPING.get(target)

    # Check by string name (for method calls)
    if isinstance(target, str):
        # Handle method calls like "relu", "transpose", etc.
        method_mapping = {
            "relu": OperationType.RELU,
            "sigmoid": OperationType.SIGMOID,
            "tanh": OperationType.TANH,
            "transpose": OperationType.TRANSPOSE,
            "permute": OperationType.PERMUTE,
            "reshape": OperationType.RESHAPE,
            "flatten": OperationType.FLATTEN,
            "sum": OperationType.SUM,
            "mean": OperationType.MEAN,
            "max": OperationType.MAX,
            "min": OperationType.MIN,
            "gather": OperationType.GATHER,
            "exp": OperationType.EXP,
            "log": OperationType.LOG,
            "sqrt": OperationType.SQRT,
            "abs": OperationType.ABS,
            "sin": OperationType.SIN,
            "cos": OperationType.COS,
            "tan": OperationType.TAN,
        }
        return method_mapping.get(target)

    return None


def is_supported_operation(target: Any) -> bool:
    """
    Check if an operation is supported for bound propagation.

    Args:
        target: The target from a torch.fx.Node

    Returns:
        True if operation is supported

    Examples:
        >>> is_supported_operation(torch.relu)
        True
        >>> is_supported_operation("custom_op")
        False
    """
    return get_operation_type(target) is not None
