"""PyTorch torch.fx-based tracing for bound propagation.

Traces a function or module into a :class:`torch.fx.GraphModule` and
validates that every operation is supported by the supplied
:class:`TargetRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

from ..propagation.registry import TargetRegistry


class BoundPropagationTracer(fx.Tracer):
    """Custom tracer with registry-based operation validation.

    Only operations registered in the provided :class:`TargetRegistry`
    are accepted.  Registered ``nn.Module`` types are kept as leaf calls;
    unregistered modules are traced into so their inner operations can be
    validated.

    Args:
        registry: The :class:`TargetRegistry` defining supported operations.
    """

    def __init__(self, registry: TargetRegistry) -> None:
        # Feed every user-callable registry target through fx's autowrap
        # mechanism so registered free functions stay as leaf call_function
        # nodes instead of being inlined. Classes (nn.Modules) are filtered
        # out — they go through is_leaf_module. Torch builtins are harmless
        # to pass (fx already treats them as leaves).
        autowrap_functions = tuple(t for t in registry.targets() if callable(t) and not isinstance(t, type))
        super().__init__(autowrap_functions=autowrap_functions)
        self._registry = registry

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """Keep standard torch.nn modules (except Sequential) and registry-registered types as leaves."""
        if isinstance(m, torch.nn.Sequential):
            return False
        if m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"):
            return True
        return self._registry.supports_target(type(m))

    def trace(
        self,
        root: Callable[..., Any] | torch.nn.Module,
        concrete_args: dict[str, Any] | None = None,
    ) -> fx.GraphModule:
        """Trace *root* and validate all operations against the registry.

        Returns:
            A :class:`torch.fx.GraphModule` ready for metadata annotation
            and bound propagation.

        Raises:
            UnsupportedOperationError: If any traced operation is not in
                the registry.
        """
        graph = super().trace(root, concrete_args=concrete_args)
        graph_module = fx.GraphModule(self.root, graph)
        self._validate_supported_operations(graph_module)
        self._validate_single_output(graph_module)
        return graph_module

    @staticmethod
    def _validate_single_output(graph_module: fx.GraphModule) -> None:
        """Reject traced graphs whose function returns a tuple/list of values.

        Bound propagation supports one output per graph; multi-output functions
        must be split into separate calls.
        """
        for node in graph_module.graph.nodes:
            if node.op != "output":
                continue
            payload = node.args[0] if node.args else None
            if isinstance(payload, (tuple, list)):
                raise MultiOutputError(
                    f"Traced function returns {len(payload)} values; only single-output "
                    f"functions are supported. Split the function or call it once per output."
                )
            return

    def _validate_supported_operations(self, graph_module: fx.GraphModule) -> None:
        """Ensure all call nodes are supported by the registry."""
        unsupported: list[str] = []

        for node in graph_module.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                if not self._registry.is_supported(node, graph_module):
                    unsupported.append(f"{node.name}: {node.op} target={node.target!r}")

        if unsupported:
            details = "\n  - " + "\n  - ".join(unsupported)
            raise UnsupportedOperationError("Traced graph contains unsupported operations:" + details)


class TraceError(Exception):
    """Base exception for tracing/validation failures."""


class UnsupportedOperationError(TraceError):
    """Raised when a traced operation has no registered strategy."""


class ControlFlowError(TraceError):
    """Raised for unsupported control flow."""


class MultiOutputError(TraceError):
    """Raised when a traced function returns multiple values."""
