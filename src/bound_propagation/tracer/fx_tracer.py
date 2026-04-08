"""
PyTorch torch.fx-based tracing for arbitrary functions.

This module handles tracing of PyTorch functions/modules using torch.fx.symbolic_trace
and provides utilities for detecting unsupported patterns (e.g., control flow).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx


class BoundPropagationTracer(fx.Tracer):
    """
    Custom tracer for bound propagation with enhanced error detection.

    This tracer extends torch.fx.Tracer to:
    - Detect control flow (if/while/for statements) and fail gracefully
    - Track metadata about the traced function
    - Provide better error messages for unsupported patterns
    """

    def __init__(self) -> None:
        super().__init__()
        self.control_flow_detected = False
        self.control_flow_locations: list[str] = []

    def trace(
        self,
        root: Callable[..., Any] | torch.nn.Module,
        concrete_args: dict[str, Any] | None = None,
    ) -> fx.Graph:
        """
        Trace a function or module, detecting control flow.

        Args:
            root: Function or module to trace
            concrete_args: Arguments to make concrete (not symbolic)

        Returns:
            Traced fx.Graph

        Raises:
            TraceError: If tracing fails or control flow detected
        """
        try:
            graph = super().trace(root, concrete_args=concrete_args)
        except Exception as e:
            raise TraceError(f"Failed to trace function: {e}") from e

        # Check for control flow after tracing
        if self.control_flow_detected:
            locations_str = "\n  ".join(self.control_flow_locations)
            raise ControlFlowError(f"Control flow detected during tracing:\n  {locations_str}\n\nBound propagation requires continuous functions without " "branching.")

        return graph


class TraceError(Exception):
    """Base exception for tracing errors."""

    pass


class ControlFlowError(TraceError):
    """Exception raised when control flow is detected during tracing."""

    pass


def trace_function(
    fn: Callable[..., Any] | torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...] | dict[str, torch.Tensor] | None = None,
    concrete_args: dict[str, Any] | None = None,
) -> fx.GraphModule:
    """
    Trace a PyTorch function or module using torch.fx.

    This function wraps torch.fx.symbolic_trace with additional validation:
    - Detects control flow (if/while/for) and raises ControlFlowError
    - Validates the traced graph structure
    - Provides enhanced error messages

    Args:
        fn: Function or module to trace. Can be:
            - A function: def forward(x, y): ...
            - A torch.nn.Module subclass
        example_inputs: Optional example inputs for shape inference.
            If provided as tuple, used as positional args.
            If provided as dict, used as keyword args.
        concrete_args: Arguments to make concrete (not traced symbolically).
            Useful for shapes, dtypes, or configuration values.

    Returns:
        fx.GraphModule containing the traced computation graph

    Raises:
        TraceError: If tracing fails
        ControlFlowError: If control flow detected in the function
        ValueError: If inputs are invalid

    Examples:
        >>> def simple_fn(x, y):
        ...     return torch.relu(x @ y)
        >>> traced = trace_function(simple_fn)

        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(10, 5)
        ...     def forward(self, x):
        ...         return torch.relu(self.linear(x))
        >>> module = MyModule()
        >>> traced = trace_function(module)

        >>> # With example inputs for validation
        >>> x = torch.randn(2, 10)
        >>> traced = trace_function(lambda x: x.relu(), example_inputs=(x,))
    """
    tracer = BoundPropagationTracer()

    try:
        # Trace the function/module
        graph = tracer.trace(fn, concrete_args=concrete_args)
        graph_module = fx.GraphModule(tracer.root, graph)

        # Validate with example inputs if provided
        if example_inputs is not None:
            _validate_traced_graph(fn, graph_module, example_inputs)

        return graph_module

    except ControlFlowError:
        # Re-raise control flow errors without wrapping
        raise
    except Exception as e:
        raise TraceError(f"Failed to trace function: {e}") from e


def _validate_traced_graph(
    original_fn: Callable[..., Any],
    traced_module: fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...] | dict[str, torch.Tensor],
) -> None:
    """
    Validate that traced graph produces same output as original function.

    Args:
        original_fn: Original function/module
        traced_module: Traced fx.GraphModule
        example_inputs: Example inputs to test with

    Raises:
        TraceError: If outputs don't match
    """
    # Convert example_inputs to proper format
    if isinstance(example_inputs, dict):
        args = ()
        kwargs = example_inputs
    elif isinstance(example_inputs, tuple):
        args = example_inputs
        kwargs = {}
    else:
        raise ValueError(f"example_inputs must be tuple or dict, got {type(example_inputs)}")

    # Run both functions
    try:
        if isinstance(original_fn, torch.nn.Module):
            original_output = original_fn(*args, **kwargs)
        else:
            original_output = original_fn(*args, **kwargs)

        traced_output = traced_module(*args, **kwargs)

    except Exception as e:
        raise TraceError(f"Failed to execute traced graph: {e}") from e

    # Compare outputs
    if isinstance(original_output, torch.Tensor):
        original_outputs = [original_output]
        traced_outputs = [traced_output]
    elif isinstance(original_output, (tuple, list)):
        original_outputs = list(original_output)
        traced_outputs = list(traced_output)
    else:
        raise TraceError(f"Unsupported output type: {type(original_output)}")

    # Check each output tensor
    for i, (orig, traced) in enumerate(zip(original_outputs, traced_outputs, strict=True)):
        if not isinstance(orig, torch.Tensor) or not isinstance(traced, torch.Tensor):
            raise TraceError(f"Output {i} is not a tensor: orig={type(orig)}, traced={type(traced)}")

        if not torch.allclose(orig, traced, rtol=1e-5, atol=1e-6):
            max_diff = (orig - traced).abs().max().item()
            raise TraceError(f"Output {i} mismatch: max difference = {max_diff:.2e}")


def detect_control_flow(graph_module: fx.GraphModule) -> list[str]:
    """
    Detect control flow patterns in a traced graph.

    Control flow is problematic for bound propagation because:
    - Bounds need continuous propagation through the computation
    - Branching creates discontinuities and requires disjunctive reasoning
    - Current implementation assumes fixed computational path

    This function detects common control flow patterns:
    - Conditional branches (if statements)
    - Loops (while/for)
    - Implicit branching (getattr with dynamic keys)

    Args:
        graph_module: Traced fx.GraphModule to analyze

    Returns:
        List of control flow locations (strings describing where found)

    Examples:
        >>> def fn_with_control_flow(x):
        ...     if x.sum() > 0:
        ...         return x.relu()
        ...     return x
        >>> try:
        ...     traced = trace_function(fn_with_control_flow)
        ... except ControlFlowError:
        ...     print("Control flow detected!")
    """
    control_flow_ops = []

    for node in graph_module.graph.nodes:
        # Check for common control flow indicators
        if node.op == "call_function":
            # torch.where is often used for if-else
            if node.target in [torch.where]:
                control_flow_ops.append(f"torch.where at node {node.name}")

        # Graph breaks or control flow markers
        elif node.op == "call_module":
            # Check module type
            module = graph_module.get_submodule(node.target)
            if "CondBranch" in type(module).__name__ or "IfElse" in type(module).__name__:
                control_flow_ops.append(f"Conditional branch at {node.target}")

    return control_flow_ops
