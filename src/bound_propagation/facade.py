"""User-facing facade for bound propagation.

Wraps the tracer, metadata pass, and propagator construction behind a
small, ergonomic API. See :class:`BoundModel` for details.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.fx as fx

from .bounds import AbstractBounds
from .passes import MetadataPass, SimplificationPass
from .propagation import (
    AlphaOptimizationConfig,
    BackwardLBPPropagator,
    BoundPropagator,
    CROWNIBPPropagator,
    ForwardBackwardLBPPropagator,
    ForwardLBPPropagator,
    IBPPropagator,
    TargetRegistry,
)
from .propagation.backward_lbp import BackwardLBPStrategy, create_default_backward_lbp_registry
from .propagation.forward_lbp import ForwardLBPStrategy, create_default_forward_lbp_registry
from .propagation.ibp import ForwardIBPStrategy, create_default_ibp_registry
from .regions import SimpleRegion
from .tracer import BoundPropagationTracer

__all__ = ["BoundModel", "Method", "RegistryExtension"]


Method = Literal["ibp", "forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"]

_REGISTRY_KEYS = ("ibp", "forward_lbp", "backward_lbp")

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
    """ANSI color helpers for :meth:`BoundModel.__repr__`.

    Colors are only applied when the output stream looks like a terminal
    that wants color: ``FORCE_COLOR`` wins, ``NO_COLOR`` disables, otherwise
    we defer to ``sys.stdout.isatty()``.
    """

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


_METHOD_REGISTRY_KEYS: dict[Method, tuple[str, ...]] = {
    "ibp": ("ibp",),
    "forward_lbp": ("forward_lbp",),
    "backward_lbp": ("backward_lbp",),
    "forward_backward_lbp": ("forward_lbp", "backward_lbp"),
    "crown_ibp": ("ibp", "backward_lbp"),
}


class _IntersectionRegistry(TargetRegistry):
    """Registry whose support is the intersection of several registries.

    Used during tracing for hybrid methods (e.g. ``crown_ibp``) so that any
    op missing from *any* required registry is rejected up front, rather
    than only at propagation time.
    """

    def __init__(self, registries: Sequence[TargetRegistry]) -> None:
        super().__init__()
        if not registries:
            raise ValueError("_IntersectionRegistry requires at least one registry")
        self._registries = tuple(registries)

    def supports_target(self, target: Callable[..., Any] | type) -> bool:
        return all(r.supports_target(target) for r in self._registries)

    def is_supported(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        return all(r.is_supported(node, graph_module) for r in self._registries)


@dataclass
class RegistryExtension:
    """Bundle of strategies for one or more fx targets, across methods.

    When a :class:`BoundModel` is built, each extension's strategies are
    registered into the registries that the chosen method requires. If the
    method needs a strategy the extension does not provide (e.g. ``method=
    "crown_ibp"`` but ``ibp=None``), construction fails with a clear error.
    Strategies for methods the chosen ``method`` doesn't use are ignored.

    Attributes
    ----------
    targets : sequence of callable or type
        The fx targets this extension applies to (e.g. ``[torch.my_op]``,
        ``[MyModule]``).
    ibp, forward_lbp, backward_lbp : strategy or None
        Strategy instance for each bound-propagation flavor. Supply only
        those needed for the methods you intend to use.
    """

    targets: Sequence[Callable[..., Any] | type]
    ibp: ForwardIBPStrategy | None = None
    forward_lbp: ForwardLBPStrategy | None = None
    backward_lbp: BackwardLBPStrategy | None = None

    def strategy_for(self, key: str) -> ForwardIBPStrategy | ForwardLBPStrategy | BackwardLBPStrategy | None:
        if key == "ibp":
            return self.ibp
        if key == "forward_lbp":
            return self.forward_lbp
        if key == "backward_lbp":
            return self.backward_lbp
        raise ValueError(f"Unknown registry key {key!r}; expected one of {_REGISTRY_KEYS}")


class BoundModel:
    """High-level facade for bound propagation.

    Traces ``model`` once at construction time, runs the metadata pass with
    the supplied ``dummy_inputs``, and builds a propagator for the chosen
    ``method``. Subsequent :meth:`propagate` calls are cheap — only the
    per-call propagator context is rebuilt.

    Parameters
    ----------
    model : callable or nn.Module
        The function or module whose bounds will be propagated.
    method : {"ibp", "forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"}
        Propagation algorithm. ``"forward_backward_lbp"`` and ``"crown_ibp"``
        use two registries internally.
    dummy_inputs : tuple of tensors
        Concrete example tensors, one per placeholder of ``model``. Used to
        run :class:`~bound_propagation.passes.MetadataPass` which annotates
        the graph with shapes, dtypes, and abstractness.
    registry : TargetRegistry or mapping, optional
        Full override of the default registries. For single-registry methods
        (``ibp``, ``forward_lbp``, ``backward_lbp``), pass a
        :class:`~bound_propagation.propagation.TargetRegistry` or a
        single-entry mapping. For dual-registry methods
        (``forward_backward_lbp``, ``crown_ibp``), pass a mapping with the
        required keys (see :attr:`BoundModel.required_registry_keys`).
        Omit to use the built-in defaults.
    extensions : sequence of RegistryExtension, optional
        Additional per-target strategies merged into whichever registries
        are in effect (defaults or user-supplied). Every extension must
        provide strategies for all of the method's required registry keys.
    alpha : AlphaOptimizationConfig, optional
        Alpha-CROWN optimization config. Only meaningful for methods whose
        propagators accept one (all LBP-based methods).
    simplify : bool, optional
        Run :class:`~bound_propagation.passes.SimplificationPass` on the
        traced graph before building the propagator. Default ``False``.
        Enable to fold algebraic identities and drop structural no-ops;
        note that some rewrites (e.g. ``x * x → pow(x, 2)``) introduce
        targets that require corresponding registry entries.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``dummy_inputs`` does not match the
        number of placeholders, a supplied registry is missing a required
        key, or an extension is missing a strategy required by ``method``.
    """

    def __init__(
        self,
        model: Callable[..., Any] | torch.nn.Module,
        dummy_inputs: Sequence[torch.Tensor],
        method: Method,
        *,
        registry: TargetRegistry | Mapping[str, TargetRegistry] | None = None,
        extensions: Sequence[RegistryExtension] = (),
        alpha: AlphaOptimizationConfig | None = None,
        simplify: bool = False,
    ) -> None:
        if method not in _METHOD_REGISTRY_KEYS:
            raise ValueError(f"Unknown method {method!r}; expected one of {tuple(_METHOD_REGISTRY_KEYS)}")

        required_keys = _METHOD_REGISTRY_KEYS[method]
        registries = self._resolve_registries(method, required_keys, registry)
        self._apply_extensions(method, required_keys, registries, extensions)

        dummy_inputs = tuple(dummy_inputs)
        tracer_registry: TargetRegistry = (
            registries[required_keys[0]]
            if len(required_keys) == 1
            else _IntersectionRegistry([registries[k] for k in required_keys])
        )
        tracer = BoundPropagationTracer(tracer_registry)
        graph_module = tracer.trace(model)

        num_placeholders = sum(1 for node in graph_module.graph.nodes if node.op == "placeholder")
        if len(dummy_inputs) != num_placeholders:
            raise ValueError(
                f"dummy_inputs has {len(dummy_inputs)} tensor(s) but the traced graph "
                f"has {num_placeholders} placeholder(s)"
            )

        # Initial MetadataPass at feature shapes validates that the model runs
        # with these shapes. It is re-run per propagate() call with batch-promoted
        # shapes so node.meta["tensor_meta"] matches the actual input rank.
        MetadataPass(graph_module).run(*dummy_inputs)

        if simplify:
            SimplificationPass().run(graph_module)
            # Rewrites invalidate meta on changed nodes; refresh so downstream
            # consumers (propagator construction, output-meta extraction) see
            # consistent shape/dtype annotations.
            MetadataPass(graph_module).run(*dummy_inputs)

        placeholder_feature_shapes = tuple(tuple(t.shape) for t in dummy_inputs)
        placeholder_dtypes = tuple(t.dtype for t in dummy_inputs)
        output_feature_shape, output_dtype = self._extract_output_meta(graph_module)

        propagator = self._build_propagator(method, graph_module, registries, alpha)

        self._method: Method = method
        self._registries = registries
        self._graph_module = graph_module
        self._propagator = propagator
        self._num_placeholders = num_placeholders
        self._placeholder_feature_shapes = placeholder_feature_shapes
        self._placeholder_dtypes = placeholder_dtypes
        self._output_feature_shape = output_feature_shape
        self._output_dtype = output_dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, *input_regions: SimpleRegion) -> AbstractBounds:
        """Propagate bounds for the given input regions.

        The number of batch dimensions is inferred per call: each region's
        shape must end with the corresponding placeholder's feature shape
        (as given by ``dummy_inputs`` at construction), and any extra
        leading dims are treated as shared batch dims. All regions must
        agree on the batch-dim count.

        Parameters
        ----------
        *input_regions : SimpleRegion
            One region per placeholder, in the same order as the traced
            model's signature.

        Returns
        -------
        AbstractBounds
            Bounds for the model's single output. Bound-propagation
            requires models to have exactly one output.
        """
        if len(input_regions) != self._num_placeholders:
            raise ValueError(f"Expected {self._num_placeholders} input region(s), got {len(input_regions)}")
        batch_ndim = self._infer_batch_ndim(input_regions)
        self._refresh_metadata(input_regions, batch_ndim)
        return self._propagator.propagate(list(input_regions), batch_ndim=batch_ndim)

    def _refresh_metadata(self, input_regions: Sequence[SimpleRegion], batch_ndim: int) -> None:
        """Re-run :class:`MetadataPass` with a zero-sample matching each region's full shape.

        Backward-mode strategies read ``node.meta["tensor_meta"]["shape"]`` and
        slice off ``batch_ndim`` leading dims to get feature shapes; this pass
        ensures the recorded shapes include the caller's actual batch dims.
        """
        if batch_ndim == 0:
            # Metadata from __init__ already matches feature shape; skip.
            return
        sample_inputs = tuple(
            torch.zeros(tuple(region.shape), dtype=dtype)
            for region, dtype in zip(input_regions, self._placeholder_dtypes, strict=True)
        )
        MetadataPass(self._graph_module).run(*sample_inputs)

    def _infer_batch_ndim(self, input_regions: Sequence[SimpleRegion]) -> int:
        """Infer the shared batch-dim count from region shapes vs. feature shapes."""
        batch_ndims: list[int] = []
        for idx, (region, feature_shape) in enumerate(
            zip(input_regions, self._placeholder_feature_shapes, strict=True)
        ):
            region_shape = tuple(region.shape)
            if len(region_shape) < len(feature_shape):
                raise ValueError(
                    f"input_regions[{idx}] has shape {region_shape} which is shorter than the "
                    f"expected feature shape {feature_shape} (from dummy_inputs)"
                )
            split = len(region_shape) - len(feature_shape)
            if region_shape[split:] != feature_shape:
                raise ValueError(
                    f"input_regions[{idx}] shape {region_shape} does not end with the expected "
                    f"feature shape {feature_shape} (from dummy_inputs)"
                )
            batch_ndims.append(split)

        if len(set(batch_ndims)) > 1:
            raise ValueError(
                f"All input regions must share the same batch-dim count; got {batch_ndims} "
                f"for feature shapes {list(self._placeholder_feature_shapes)}"
            )
        return batch_ndims[0] if batch_ndims else 0

    def __repr__(self) -> str:
        c = _Palette.resolve()
        indent = "  "
        lines = [f"{type(self).__name__}("]
        lines.append(f"{indent}method:     {c.name(self._method)}")
        registries = ", ".join(c.name(k) for k in self.required_registry_keys)
        lines.append(f"{indent}registries: ({registries})")

        lines.append(f"{indent}inputs:")
        for idx, (shape, dtype) in enumerate(
            zip(self._placeholder_feature_shapes, self._placeholder_dtypes, strict=True)
        ):
            lines.append(f"{indent * 2}[{idx}] shape={c.shape(shape)} dtype={c.dtype(dtype)}")

        out_shape = c.shape(self._output_feature_shape) if self._output_feature_shape is not None else c.dim("?")
        out_dtype = c.dtype(self._output_dtype) if self._output_dtype is not None else c.dim("?")
        lines.append(f"{indent}output:     shape={out_shape} dtype={out_dtype}")

        op_counts = Counter(node.op for node in self._graph_module.graph.nodes)
        total = sum(op_counts.values())
        lines.append(f"{indent}graph:      {c.count(total)} nodes")
        width = max(len(op) for op in op_counts)
        for op, count in sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"{indent * 2}{op.ljust(width)}  {c.count(count)}")

        lines.append(")")
        return "\n".join(lines)

    @property
    def method(self) -> Method:
        return self._method

    @property
    def graph_module(self) -> fx.GraphModule:
        return self._graph_module

    @property
    def propagator(self) -> BoundPropagator:
        return self._propagator

    @property
    def registries(self) -> Mapping[str, TargetRegistry]:
        return self._registries

    @property
    def required_registry_keys(self) -> tuple[str, ...]:
        return _METHOD_REGISTRY_KEYS[self._method]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_meta(
        graph_module: fx.GraphModule,
    ) -> tuple[tuple[int, ...] | None, torch.dtype | None]:
        output_node = next(n for n in graph_module.graph.nodes if n.op == "output")
        value_node = output_node.args[0]
        if not isinstance(value_node, fx.Node):
            return None, None
        meta = value_node.meta.get("tensor_meta")
        if meta is None:
            return None, None
        return tuple(meta["shape"]), meta.get("dtype")

    @staticmethod
    def _resolve_registries(
        method: Method,
        required_keys: tuple[str, ...],
        registry: TargetRegistry | Mapping[str, TargetRegistry] | None,
    ) -> dict[str, TargetRegistry]:
        defaults: dict[str, Callable[[], TargetRegistry]] = {
            "ibp": create_default_ibp_registry,
            "forward_lbp": create_default_forward_lbp_registry,
            "backward_lbp": create_default_backward_lbp_registry,
        }

        if registry is None:
            return {key: defaults[key]() for key in required_keys}

        if isinstance(registry, TargetRegistry):
            if len(required_keys) != 1:
                raise ValueError(
                    f"method={method!r} needs registries for {required_keys}; pass a mapping "
                    f"(e.g. {{'forward_lbp': ..., 'backward_lbp': ...}}) instead of a single registry"
                )
            resolved = {required_keys[0]: registry}
            return resolved

        if isinstance(registry, Mapping):
            missing = [k for k in required_keys if k not in registry]
            if missing:
                raise ValueError(f"method={method!r} requires registry keys {required_keys}; missing {missing}")
            unexpected = [k for k in registry if k not in required_keys]
            if unexpected:
                raise ValueError(f"method={method!r} expects only keys {required_keys}; got extra {unexpected}")
            return {k: registry[k] for k in required_keys}

        raise TypeError(f"registry must be a TargetRegistry, a mapping, or None; got {type(registry).__name__}")

    @staticmethod
    def _apply_extensions(
        method: Method,
        required_keys: tuple[str, ...],
        registries: dict[str, TargetRegistry],
        extensions: Sequence[RegistryExtension],
    ) -> None:
        for idx, ext in enumerate(extensions):
            if not ext.targets:
                raise ValueError(f"extensions[{idx}] has no targets")
            for key in required_keys:
                strategy = ext.strategy_for(key)
                if strategy is None:
                    raise ValueError(
                        f"extensions[{idx}] is missing the {key!r} strategy required by "
                        f"method={method!r} (targets={list(ext.targets)!r})"
                    )
                registries[key].register_many(list(ext.targets), strategy)

    @staticmethod
    def _build_propagator(
        method: Method,
        graph_module: fx.GraphModule,
        registries: dict[str, TargetRegistry],
        alpha: AlphaOptimizationConfig | None,
    ) -> BoundPropagator:
        if method == "ibp":
            if alpha is not None and alpha.enabled:
                raise ValueError("IBP does not support alpha-CROWN optimization; pass alpha=None")
            return IBPPropagator(graph_module, registry=registries["ibp"])
        if method == "forward_lbp":
            return ForwardLBPPropagator(
                graph_module,
                registry=registries["forward_lbp"],
                alpha_config=alpha,
            )
        if method == "backward_lbp":
            return BackwardLBPPropagator(
                graph_module,
                registry=registries["backward_lbp"],
                alpha_config=alpha,
            )
        if method == "forward_backward_lbp":
            return ForwardBackwardLBPPropagator(
                graph_module,
                forward_registry=registries["forward_lbp"],
                backward_registry=registries["backward_lbp"],
                alpha_config=alpha,
            )
        if method == "crown_ibp":
            return CROWNIBPPropagator(
                graph_module,
                ibp_registry=registries["ibp"],
                backward_registry=registries["backward_lbp"],
                alpha_config=alpha,
            )
        raise ValueError(f"Unhandled method {method!r}")
