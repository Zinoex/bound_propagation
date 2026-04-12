"""PyTorch torch.fx-based tracing for arbitrary functions.

This tracer enforces an operation allowlist based on the operation mapping used by
the IR converter. Any operation that cannot be mapped is rejected at trace-time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

from .op_mapping import MODULE_OP_MAPPING, is_supported_operation


class BoundPropagationTracer(fx.Tracer):
    """
    Custom tracer for bound propagation with mapped-operation enforcement.

    This tracer extends torch.fx.Tracer to:
    - Allow only operations present in the tracer/converter op mapping
    - Keep mapped modules as leaf calls, while tracing through unmapped modules
    - Fail early with actionable error messages
    """

    def __init__(self) -> None:
        super().__init__()

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        Keep only mapped module types as leaf modules.

        Unmapped modules are traced into so their inner operations can be checked
        against the same operation allowlist.
        """
        del module_qualified_name
        return type(m) in MODULE_OP_MAPPING

    def trace(
        self,
        root: Callable[..., Any] | torch.nn.Module,
        concrete_args: dict[str, Any] | None = None,
    ) -> fx.Graph:
        """
        Trace a function or module and reject unsupported operations.

        Args:
            root: Function or module to trace
            concrete_args: Arguments to make concrete (not symbolic)

        Returns:
            Traced fx.Graph

        Raises:
            UnsupportedOperationError: If any traced op is not mapped
        """
        graph = super().trace(root, concrete_args=concrete_args)
        self._validate_supported_operations(graph)

        return graph

    def _validate_supported_operations(self, graph: fx.Graph) -> None:
        """Ensure all call nodes correspond to mapped operations."""
        unsupported_nodes: list[str] = []

        for node in graph.nodes:
            if node.op in {"call_function", "call_method"}:
                if not is_supported_operation(node.target):
                    unsupported_nodes.append(f"{node.name}: {node.op} target={node.target!r}")

            elif node.op == "call_module":
                module_type = self._get_module_type(node)
                if module_type is None or module_type not in MODULE_OP_MAPPING:
                    unsupported_nodes.append(f"{node.name}: call_module target={node.target!r} type={module_type}")

        if unsupported_nodes:
            details = "\n  - " + "\n  - ".join(unsupported_nodes)
            raise UnsupportedOperationError("Traced graph contains unsupported operations. Only mapped operations are accepted:" + details)

    def _get_module_type(self, node: fx.Node) -> type[torch.nn.Module] | None:
        """Resolve module type for a call_module node."""
        if not isinstance(self.root, torch.nn.Module):
            return None

        try:
            module = self.root.get_submodule(str(node.target))
        except Exception:
            return None

        return type(module)


class TraceError(Exception):
    """Base exception raised for tracing/validation failures."""

    ...


class UnsupportedOperationError(TraceError):
    """Exception raised when an operation not present in op mapping is traced."""

    ...


class ControlFlowError(Exception):
    """Backward-compatible alias for legacy control-flow related errors."""

    ...
