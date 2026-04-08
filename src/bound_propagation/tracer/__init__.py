"""
Graph tracing and conversion to IR.

This module provides functionality to trace PyTorch functions using torch.fx
and convert them to the internal IR representation.
"""

from .converter import GraphConverter
from .fx_tracer import trace_function

__all__ = [
    "trace_function",
    "GraphConverter",
]
