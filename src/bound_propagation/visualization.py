"""Optional matplotlib-based visualization for bound-propagation sanity checks.

Plots the true function alongside IBP and (forward / backward) LBP bounds on a
1D input region. Useful for visually verifying soundness and tightness across
operations and small compositions.

Matplotlib is a soft dependency: install via ``pip install bound_propagation[viz]``
or ``pip install matplotlib``. The plotting functions raise a clear
``ImportError`` if matplotlib is missing.

Scope
-----
- Input region must be 1-dimensional (``region.shape == (1,)``).
- Function output may be scalar or vector; for vectors, ``output_index``
  selects which component to plot.
- The three default methods (``ibp``, ``forward_lbp``, ``backward_lbp``) are
  attempted independently; if a method fails on a particular operation, the
  failure is reported on the axes and the other methods still render.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from .bounds import AbstractBounds, IntervalBounds, LinearBounds
from .facade import BoundModel, Method
from .regions import HyperRectangle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


__all__ = ["DEFAULT_METHODS", "plot_bounds"]


DEFAULT_METHODS: tuple[Method, ...] = ("ibp", "forward_lbp", "backward_lbp")

_METHOD_COLORS: dict[Method, str] = {
    "ibp": "tab:red",
    "forward_lbp": "tab:blue",
    "backward_lbp": "tab:green",
    "forward_backward_lbp": "tab:purple",
    "crown_ibp": "tab:orange",
}


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install matplotlib` "
            "or as an optional extra: `pip install bound_propagation[viz]`."
        ) from exc
    return plt


def plot_bounds(
    fn: Callable[..., torch.Tensor] | torch.nn.Module,
    region: HyperRectangle,
    *,
    methods: Sequence[Method] = DEFAULT_METHODS,
    num_samples: int = 200,
    output_index: int = 0,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot bounds against the true function on a 1D input region.

    Parameters
    ----------
    fn : Callable or nn.Module
        The function under test. Must accept a 1D tensor input of shape ``(1,)``.
    region : HyperRectangle
        Input region. ``region.shape`` must be ``(1,)``.
    methods : sequence of Method
        Which bound-propagation methods to plot. Defaults to all three single-
        registry methods. Methods that fail on this op are reported in-axes
        and the others still render.
    num_samples : int
        Number of x-samples used to draw the true function curve.
    output_index : int
        For vector-valued ``fn``, which output component to plot (``f(x).flatten()[output_index]``).
    title : str, optional
        Axis title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If ``None``, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``region`` is not 1-dimensional.
    """
    plt = _require_matplotlib()

    if tuple(region.shape) != (1,):
        raise ValueError(f"plot_bounds requires a 1D input region (shape (1,)), got shape {tuple(region.shape)}")

    lo = float(region.lower[0])
    hi = float(region.upper[0])
    xs = torch.linspace(lo, hi, num_samples)
    true_ys = _evaluate_function(fn, xs, output_index)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
    else:
        # ``ax.figure`` is typed as ``Figure | SubFigure``; users supply a top-level
        # axes here so the runtime value is always ``Figure``.
        fig = ax.figure

    ax.plot(xs.numpy(), true_ys.numpy(), color="black", linewidth=2.0, label="f(x)", zorder=10)

    dummy = torch.zeros(1)
    for method in methods:
        color = _METHOD_COLORS.get(method, None)
        try:
            model = BoundModel(fn, dummy_inputs=(dummy,), method=method)
            bounds = model.propagate(region)
        except Exception as exc:  # noqa: BLE001
            _annotate_failure(ax, method, exc, color=color)
            continue
        _draw_bound(ax, bounds, xs, output_index=output_index, label=method, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    return fig  # ty:ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _evaluate_function(
    fn: Callable[..., torch.Tensor] | torch.nn.Module,
    xs: torch.Tensor,
    output_index: int,
) -> torch.Tensor:
    """Evaluate ``fn`` at each scalar sample in ``xs`` and pick component ``output_index``."""
    ys: list[torch.Tensor] = []
    with torch.no_grad():
        for x in xs:
            y = fn(x.reshape(1))
            if not isinstance(y, torch.Tensor):
                raise TypeError(f"fn must return torch.Tensor; got {type(y).__name__}")
            ys.append(y.flatten()[output_index].detach())
    return torch.stack(ys)


def _draw_bound(
    ax: Axes,
    bounds: AbstractBounds,
    xs: torch.Tensor,
    *,
    output_index: int,
    label: str,
    color: str | None,
) -> None:
    if isinstance(bounds, IntervalBounds):
        flat_lower = bounds.lower.detach().flatten()
        flat_upper = bounds.upper.detach().flatten()
        lower_val = float(flat_lower[output_index])
        upper_val = float(flat_upper[output_index])
        ax.fill_between(xs.numpy(), lower_val, upper_val, alpha=0.12, color=color, label=label)
        return
    if isinstance(bounds, LinearBounds):
        b_lo = float(bounds.bias_lower.detach().flatten()[output_index])
        b_up = float(bounds.bias_upper.detach().flatten()[output_index])
        if not bounds.has_linear_terms():
            # Constant bound (e.g. produced when an upstream op chain-breaks
            # to interval bounds). Draw as a horizontal band like IBP.
            ax.fill_between(xs.numpy(), b_lo, b_up, alpha=0.12, color=color, label=f"{label} (const)")
            return
        coeff = bounds.coefficient
        output_size = int(torch.tensor(bounds.bias_lower.shape).prod().item())
        lower_dense = coeff.lower.to_dense().tensor.detach().reshape(output_size, -1)
        upper_dense = coeff.upper.to_dense().tensor.detach().reshape(output_size, -1)
        # 1D input region: input_size == 1.
        a_lo = float(lower_dense[output_index, 0])
        a_up = float(upper_dense[output_index, 0])
        ys_lo = a_lo * xs + b_lo
        ys_up = a_up * xs + b_up
        ax.plot(xs.numpy(), ys_lo.numpy(), linestyle="--", color=color, alpha=0.85, label=f"{label} L")
        ax.plot(xs.numpy(), ys_up.numpy(), linestyle=":", color=color, alpha=0.85, label=f"{label} U")
        return
    raise TypeError(f"Unexpected bound type {type(bounds).__name__}")


def _annotate_failure(ax: Axes, method: str, exc: Exception, *, color: str | None) -> None:
    """Note that ``method`` failed without crashing the rest of the plot."""
    msg = f"{method}: {exc.__class__.__name__}"
    # Stash the failure as an empty-data line so it appears in the legend with the method's color.
    ax.plot([], [], color=color, linestyle="-", alpha=0.5, label=msg)
