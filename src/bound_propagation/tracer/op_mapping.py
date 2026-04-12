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
    # Elementwise arithmetic
    torch.add: OperationType.ADD,
    torch.Tensor.add: OperationType.ADD,
    operator.add: OperationType.ADD,
    torch.sub: OperationType.SUB,
    torch.Tensor.sub: OperationType.SUB,
    operator.sub: OperationType.SUB,
    torch.mul: OperationType.MUL,
    torch.Tensor.mul: OperationType.MUL,
    operator.mul: OperationType.MUL,
    torch.div: OperationType.DIV,
    torch.Tensor.div: OperationType.DIV,
    operator.truediv: OperationType.DIV,
    torch.pow: OperationType.POW,
    torch.Tensor.pow: OperationType.POW,
    operator.pow: OperationType.POW,
    torch.neg: OperationType.NEG,
    torch.Tensor.neg: OperationType.NEG,
    operator.neg: OperationType.NEG,
    torch.reciprocal: OperationType.RECIPROCAL,
    torch.Tensor.reciprocal: OperationType.RECIPROCAL,
    torch.maximum: OperationType.MAXIMUM,
    torch.minimum: OperationType.MINIMUM,
    # Activations
    torch.relu: OperationType.RELU,
    torch.Tensor.relu: OperationType.RELU,
    F.relu: OperationType.RELU,
    torch.sigmoid: OperationType.SIGMOID,
    torch.Tensor.sigmoid: OperationType.SIGMOID,
    F.sigmoid: OperationType.SIGMOID,
    torch.tanh: OperationType.TANH,
    torch.Tensor.tanh: OperationType.TANH,
    F.tanh: OperationType.TANH,
    torch.Tensor.exp: OperationType.EXP,
    torch.exp: OperationType.EXP,
    torch.log: OperationType.LOG,
    torch.Tensor.log: OperationType.LOG,
    torch.sqrt: OperationType.SQRT,
    torch.Tensor.sqrt: OperationType.SQRT,
    torch.abs: OperationType.ABS,
    torch.Tensor.abs: OperationType.ABS,
    torch.clamp: OperationType.CLAMP,
    torch.Tensor.clamp: OperationType.CLAMP,
    # Trigonometric
    torch.sin: OperationType.SIN,
    torch.Tensor.sin: OperationType.SIN,
    torch.cos: OperationType.COS,
    torch.Tensor.cos: OperationType.COS,
    torch.tan: OperationType.TAN,
    torch.Tensor.tan: OperationType.TAN,
    # Reductions
    torch.sum: OperationType.SUM,
    torch.Tensor.sum: OperationType.SUM,
    torch.mean: OperationType.MEAN,
    torch.Tensor.mean: OperationType.MEAN,
    torch.max: OperationType.MAX,
    torch.Tensor.max: OperationType.MAX,
    torch.min: OperationType.MIN,
    torch.Tensor.min: OperationType.MIN,
    # Structural
    torch.cat: OperationType.CONCAT,
    torch.concat: OperationType.CONCAT,
    torch.stack: OperationType.STACK,
    torch.reshape: OperationType.RESHAPE,
    torch.Tensor.reshape: OperationType.RESHAPE,
    torch.flatten: OperationType.FLATTEN,
    torch.Tensor.flatten: OperationType.FLATTEN,
    torch.unsqueeze: OperationType.UNSQUEEZE,
    torch.Tensor.unsqueeze: OperationType.UNSQUEEZE,
    torch.squeeze: OperationType.SQUEEZE,
    torch.Tensor.squeeze: OperationType.SQUEEZE,
    # Indexing/selection
    operator.getitem: OperationType.GETITEM,
    torch.select: OperationType.SELECT,
    torch.Tensor.select: OperationType.SELECT,
    torch.transpose: OperationType.TRANSPOSE,
    torch.Tensor.transpose: OperationType.TRANSPOSE,
    torch.permute: OperationType.PERMUTE,
    torch.Tensor.permute: OperationType.PERMUTE,
    torch.Tensor.view: OperationType.VIEW,
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
