"""ANSI-coloured pretty-printer for :class:`BoundModel`.

Lifted out of ``facade.py`` because the ANSI logic is orthogonal to bound
propagation and was the bulk of the file. ``BoundModel.__repr__`` delegates
to :func:`format_bound_model`.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .facade import BoundModel


_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "magenta": "\x1b[35m",
}


@dataclass(frozen=True)
class _Palette:
    """ANSI colour helpers. ``FORCE_COLOR`` wins, ``NO_COLOR`` disables, otherwise tty-detection."""

    enabled: bool

    @classmethod
    def resolve(cls) -> _Palette:
        if os.environ.get("NO_COLOR"):
            return cls(enabled=False)
        if os.environ.get("FORCE_COLOR"):
            return cls(enabled=True)
        return cls(enabled=bool(getattr(sys.stdout, "isatty", lambda: False)()))

    def _wrap(self, text: str, *codes: str) -> str:
        if not self.enabled:
            return text
        prefix = "".join(_ANSI[c] for c in codes)
        return f"{prefix}{text}{_ANSI['reset']}"

    def name(self, value: Any) -> str:
        return self._wrap(str(value), "bold", "cyan")

    def shape(self, value: Any) -> str:
        return self._wrap(str(tuple(value)), "green")

    def dtype(self, value: Any) -> str:
        return self._wrap(str(value), "magenta")

    def count(self, value: Any) -> str:
        return self._wrap(str(value), "yellow")

    def dim(self, value: Any) -> str:
        return self._wrap(str(value), "dim")


def format_bound_model(model: BoundModel) -> str:
    """Pretty-print a :class:`BoundModel` with method, registries, IO shapes, and graph counts."""
    c = _Palette.resolve()
    indent = "  "
    lines = [f"{type(model).__name__}("]
    lines.append(f"{indent}method:     {c.name(model.method)}")
    registries = ", ".join(c.name(k) for k in model.required_registry_keys)
    lines.append(f"{indent}registries: ({registries})")

    lines.append(f"{indent}inputs:")
    for idx, (shape, dtype) in enumerate(
        zip(model._placeholder_feature_shapes, model._placeholder_dtypes, strict=True)
    ):
        lines.append(f"{indent * 2}[{idx}] shape={c.shape(shape)} dtype={c.dtype(dtype)}")

    out_shape = c.shape(model._output_feature_shape) if model._output_feature_shape is not None else c.dim("?")
    out_dtype = c.dtype(model._output_dtype) if model._output_dtype is not None else c.dim("?")
    lines.append(f"{indent}output:     shape={out_shape} dtype={out_dtype}")

    op_counts = Counter(node.op for node in model._graph_module.graph.nodes)
    total = sum(op_counts.values())
    lines.append(f"{indent}graph:      {c.count(total)} nodes")
    width = max(len(op) for op in op_counts)
    for op, count in sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{indent * 2}{op.ljust(width)}  {c.count(count)}")

    lines.append(")")
    return "\n".join(lines)
