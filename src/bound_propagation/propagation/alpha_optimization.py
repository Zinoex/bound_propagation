"""Infrastructure for optimizable linear relaxations (alpha-CROWN).

This module defines the core abstractions used to make the free parameters of
linear relaxations (ReLU crossing slope, sigmoid tangent points, McCormick eta
weights, etc.) learnable via gradient descent.

Central invariant
-----------------
Every optimizable knob, regardless of operation, is represented as a tensor of
unit-interval fractions ``alpha in [0, 1]``. Each ``compute_*_relaxation``
function is responsible for mapping the fraction to the geometric quantity it
parameterizes (slope, tangent point x-coordinate, McCormick weight). This
keeps the optimizer op-agnostic: a single ``clamp_(0, 1)`` is the universal
projection step.

Key types
---------
- :class:`AlphaOptimizationConfig`: user-facing dataclass controlling the
  optimization loop (enabled, iterations, lr, scope, loss, optimizer).
- :class:`AlphaProvider`: protocol that strategies call to obtain an
  optimizable override for a specific ``(node, knob)`` pair.
- :class:`AlphaStore`: ``nn.Module`` holding lazily-allocated parameters,
  keyed by ``(node_name, knob_name)``. Parameters are always stored as
  unit-interval fractions and are clamped in-place by :meth:`project`.
- :class:`AutoRegisteringAlphaProvider`: concrete provider that lazily grows
  :class:`AlphaStore` as strategies request new knobs during a forward walk.
- :class:`NullAlphaProvider`: sentinel provider used during warm-up passes or
  explicit "no optimization" calls; always returns ``None``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.fx as fx
import torch.nn as nn

from ..bounds import LinearBounds


@dataclass(frozen=True)
class AlphaOptimizationConfig:
    """Configuration for optimizable linear relaxations.

    Attributes
    ----------
    enabled : bool
        Master switch. When ``False``, the propagator runs a single forward
        walk with the analytical defaults (no learnable parameters, no
        optimization loop).
    iterations : int
        Number of projected-gradient-descent steps to run. Ignored when
        ``enabled`` is ``False``.
    lr : float
        Learning rate for the outer optimizer.
    optimize_intermediate : bool
        When ``True`` (only valid for :class:`BackwardLBPPropagator`), the
        intermediate bound concretizations inside the backward tape flow
        gradients through their alphas — every layer's knobs couple through
        downstream interval bounds. When ``False``, intermediate
        concretizations are wrapped in ``torch.no_grad()`` and only the
        final output's backward pass receives gradients.
    loss : {"width", "lower", "upper"}
        Scalar objective derived from the concretized output bounds. ``width``
        minimizes ``sum(upper) - sum(lower)``; ``lower`` maximizes
        ``sum(lower)``; ``upper`` minimizes ``sum(upper)``.
    optimizer_name : {"adam", "sgd"}
        Outer optimizer choice. ``adam`` is the alpha-CROWN default.
    """

    enabled: bool = False
    iterations: int = 20
    lr: float = 0.1
    optimize_intermediate: bool = False
    loss: Literal["width", "lower", "upper"] = "width"
    optimizer_name: Literal["adam", "sgd"] = "adam"


class AlphaProvider(Protocol):
    """Protocol for resolving optimizable alpha overrides during propagation.

    A strategy that supports alpha optimization calls :meth:`get` with a
    unique ``(node, knob_name)`` key, the shape of the tensor it wants,
    and the per-element initialization fraction(s) that would reproduce
    the op's analytical default. The method returns either:

    - a ``torch.Tensor`` of unit-interval fractions (shape matching the
      request) to use as the override; or
    - ``None`` when optimization is disabled for this key (the strategy
      then falls back to the analytical default).

    Implementations are expected to be stateful across calls within a single
    forward walk so that repeated calls with the same key return the *same*
    tensor object, allowing gradients to accumulate.
    """

    def get(
        self,
        node: fx.Node,
        knob_name: str,
        shape: torch.Size,
        init: float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Resolve the alpha override for ``(node, knob_name)``.

        Parameters
        ----------
        node : fx.Node
            The graph node whose relaxation is being built.
        knob_name : str
            An operation-local identifier for the knob (e.g.
            ``"relu_lower_slope"``, ``"sigmoid_tangent_lower"``). Must be
            unique per node per op.
        shape : torch.Size
            Shape of the override tensor (matches the regime tensors for
            the op, which are typically the shape of the node's input).
        init : float | torch.Tensor
            Initial fraction in ``[0, 1]`` reproducing the analytical
            default. Either scalar (applied element-wise) or a tensor
            broadcastable to ``shape``. Only consulted on first allocation;
            subsequent calls with the same key ignore ``init``.
        device : torch.device
            Device the override must live on.
        dtype : torch.dtype
            Dtype the override must use.

        Returns
        -------
        torch.Tensor | None
            A unit-interval fraction tensor, or ``None`` if the provider
            decides this knob should remain at its analytical default.
        """
        ...


class NullAlphaProvider:
    """Provider that always returns ``None`` — analytical defaults everywhere.

    Use this when alpha optimization is disabled but a call site still
    expects an ``AlphaProvider``.
    """

    def get(
        self,
        node: fx.Node,
        knob_name: str,
        shape: torch.Size,
        init: float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return None


class AlphaStore(nn.Module):
    """Container for optimizable alpha parameters.

    Parameters are stored as unit-interval fractions and allocated lazily on
    first access. Each parameter is keyed by ``(node_name, knob_name)`` and
    registered under a ``<node_name>__<knob_name>`` attribute so it is
    discoverable via :meth:`nn.Module.parameters` (and therefore by standard
    ``torch.optim`` optimizers).

    The store does not track an op-specific range: everything lives in
    ``[0, 1]`` by design, and :meth:`project` enforces that bound in-place
    after each optimizer step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._index: dict[tuple[str, str], str] = {}

    @staticmethod
    def _attr_name(node_name: str, knob_name: str) -> str:
        """Sanitize ``(node_name, knob_name)`` into a valid Python attribute.

        ``fx.Node`` names are already valid identifiers; this only needs to
        combine the two parts without introducing collisions between
        dissimilar keys.
        """
        return f"{node_name}__{knob_name}"

    def get_or_create(
        self,
        node_name: str,
        knob_name: str,
        shape: torch.Size,
        init: float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the ``nn.Parameter`` for ``(node_name, knob_name)``.

        On first access allocates a parameter of ``shape`` initialized
        from ``init`` (which must already be in ``[0, 1]`` — callers are
        responsible for picking the fraction that reproduces the analytical
        default of the op).

        Parameters
        ----------
        node_name : str
            The ``fx.Node.name`` the parameter belongs to.
        knob_name : str
            Per-op identifier for the knob.
        shape : torch.Size
            Shape of the parameter tensor.
        init : float | torch.Tensor
            Initial fraction(s) in ``[0, 1]``. If a scalar, every element is
            set to ``init``. If a tensor, it must be broadcastable to
            ``shape`` and have every entry in ``[0, 1]``.
        device : torch.device
            Device to allocate the parameter on.
        dtype : torch.dtype
            Dtype for the parameter.

        Returns
        -------
        torch.Tensor
            The parameter tensor. Subsequent calls with the same key return
            the same object so that gradients accumulate correctly.
        """
        key = (node_name, knob_name)
        attr = self._index.get(key)
        if attr is not None:
            return getattr(self, attr)

        attr = self._attr_name(node_name, knob_name)
        if hasattr(self, attr):
            raise RuntimeError(
                f"AlphaStore attribute name collision for key ({node_name!r}, {knob_name!r}): "
                f"attribute {attr!r} is already defined."
            )

        if isinstance(init, torch.Tensor):
            init_tensor = init.to(device=device, dtype=dtype).expand(tuple(shape)).contiguous().detach().clone()
            if torch.any((init_tensor < 0.0) | (init_tensor > 1.0)):
                raise ValueError(
                    f"AlphaStore tensor init must be in [0, 1] for key ({node_name!r}, {knob_name!r}); "
                    f"saw min={init_tensor.min().item()!r} max={init_tensor.max().item()!r}."
                )
        else:
            if not 0.0 <= init <= 1.0:
                raise ValueError(
                    f"AlphaStore scalar init must be in [0, 1]; got init={init!r} "
                    f"for key ({node_name!r}, {knob_name!r})."
                )
            init_tensor = torch.full(tuple(shape), float(init), device=device, dtype=dtype)

        param = nn.Parameter(init_tensor)
        self.register_parameter(attr, param)
        self._index[key] = attr
        return param

    def contains(self, node_name: str, knob_name: str) -> bool:
        """Whether a parameter has been allocated for ``(node_name, knob_name)``."""
        return (node_name, knob_name) in self._index

    def keys(self) -> list[tuple[str, str]]:
        """All ``(node_name, knob_name)`` keys currently allocated."""
        return list(self._index)

    @torch.no_grad()
    def project(self) -> None:
        """Clamp every parameter in-place to the unit interval ``[0, 1]``.

        Call after each optimizer step to keep all knobs in their valid
        range (projected gradient descent).
        """
        for attr in self._index.values():
            param = getattr(self, attr)
            param.clamp_(0.0, 1.0)


class AutoRegisteringAlphaProvider:
    """Alpha provider that lazily grows an :class:`AlphaStore`.

    On the first :meth:`get` call for each ``(node, knob_name)`` key, the
    provider allocates a new parameter in the backing store using the
    supplied ``init`` value. Subsequent calls return the same parameter
    tensor (so gradients accumulate).

    When ``frozen`` is ``True``, the provider still returns already-allocated
    parameters but refuses to allocate new ones (used for the final
    post-optimization ``no_grad`` pass to guard against key drift). In that
    mode, unknown keys return ``None`` so the strategy falls back to its
    analytical default.

    Parameters
    ----------
    store : AlphaStore
        The backing store. Typically owned by the propagator for the
        duration of a single :meth:`propagate` call.
    frozen : bool
        When ``True``, only pre-existing keys are served; new keys return
        ``None``.
    """

    def __init__(self, store: AlphaStore, *, frozen: bool = False) -> None:
        self._store = store
        self._frozen = frozen

    @property
    def store(self) -> AlphaStore:
        return self._store

    def freeze(self) -> AutoRegisteringAlphaProvider:
        """Return a frozen view over the same store."""
        return AutoRegisteringAlphaProvider(self._store, frozen=True)

    def get(
        self,
        node: fx.Node,
        knob_name: str,
        shape: torch.Size,
        init: float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self._frozen and not self._store.contains(node.name, knob_name):
            return None

        return self._store.get_or_create(
            node_name=node.name,
            knob_name=knob_name,
            shape=shape,
            init=init,
            device=device,
            dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------


def _alpha_loss(
    bounds: LinearBounds,
    kind: Literal["width", "lower", "upper"],
) -> torch.Tensor:
    """Reduce a :class:`LinearBounds` to a scalar loss.

    - ``"width"``: minimize ``sum(upper) - sum(lower)``.
    - ``"lower"``: maximize ``sum(lower)`` (returned as its negation).
    - ``"upper"``: minimize ``sum(upper)``.
    """
    concrete = bounds.concretize()
    if kind == "width":
        return (concrete.upper - concrete.lower).sum()
    if kind == "lower":
        return -concrete.lower.sum()
    if kind == "upper":
        return concrete.upper.sum()
    raise ValueError(f"Unknown alpha loss kind: {kind!r}")


def _build_optimizer(store: AlphaStore, config: AlphaOptimizationConfig) -> torch.optim.Optimizer:
    """Build the outer optimizer over every parameter registered in the store."""
    params = list(store.parameters())
    if not params:
        raise RuntimeError(
            "AlphaStore has no registered parameters after the warm-up pass. "
            "Either the graph has no alpha-capable operations, or the warm-up did not consult the provider."
        )

    if config.optimizer_name == "adam":
        return torch.optim.Adam(params, lr=config.lr)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=config.lr)
    raise ValueError(f"Unknown optimizer_name: {config.optimizer_name!r}")


def run_alpha_optimization(
    propagate_once: Callable[[AlphaProvider], LinearBounds],
    config: AlphaOptimizationConfig,
) -> LinearBounds:
    """Run projected-gradient-descent on alpha parameters.

    Orchestrates the standard alpha-CROWN outer loop:

    1. **Warm-up pass** — call ``propagate_once`` with an auto-registering
       provider so every alpha-capable strategy allocates its parameter in
       a fresh :class:`AlphaStore`.
    2. **Optimization iterations** — repeatedly rebuild the tape, compute a
       scalar loss from the output bounds, backpropagate, step Adam/SGD,
       and clamp every parameter to ``[0, 1]``.
    3. **Final pass** — one last ``propagate_once`` under
       :func:`torch.no_grad` using the frozen alphas; the result is
       returned to the caller.

    Parameters
    ----------
    propagate_once : Callable[[AlphaProvider], Sequence[LinearBounds]]
        A closure that runs the propagator's forward walk + backward tape
        pass using the supplied provider to resolve alpha overrides, and
        returns the per-output bounds.
    config : AlphaOptimizationConfig
        Must have ``config.enabled == True`` (callers should gate).

    Returns
    -------
    Sequence[LinearBounds]
        The bounds produced by the final no-grad pass.
    """
    if not config.enabled:
        raise ValueError("run_alpha_optimization called with config.enabled=False")

    store = AlphaStore()
    provider = AutoRegisteringAlphaProvider(store)

    # Warm-up: let every alpha-capable strategy register its knobs.
    with torch.no_grad():
        _ = propagate_once(provider)

    optimizer = _build_optimizer(store, config)

    for _ in range(config.iterations):
        optimizer.zero_grad()
        bounds = propagate_once(provider)
        loss = _alpha_loss(bounds, config.loss)
        loss.backward()
        optimizer.step()
        store.project()

    # TODO: Consider snapping every parameter to nearest point of {0, 1, slope-0}
    # after optimization. I conjecture that the optimal solution always lies at one
    # of these extreme/key points. This will require testing to confirm.

    with torch.no_grad():
        return propagate_once(provider.freeze())
