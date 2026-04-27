"""Optional matplotlib-based visualization for bound-propagation sanity checks.

The module exposes two related entry points:

- :func:`plot_bounds` — render bounds for a single ``(fn, region)`` pair onto
  one axes. Used to verify that a chosen propagation method produces a sound
  envelope around the true function on a 1D input region.
- :func:`plot_bounds_grid` — render many ``(fn, region)`` panels into a single
  figure. Used by the showcase scripts in ``scripts/`` to sweep a curated set
  of operations / compositions.

Matplotlib is a soft dependency: install via ``pip install bound_propagation[viz]``
or ``pip install matplotlib``. The plotting functions raise a clear
``ImportError`` if matplotlib is missing.

Scope
-----
- Input region must be 1-dimensional (``region.shape == (1,)``).
- Function output may be scalar or vector; for vectors, ``output_index``
  selects which component to plot.
- Methods are attempted independently per axes; if one fails on a particular
  operation, the failure is reported on the axes and the others still render.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .bounds import AbstractBounds, IntervalBounds, LinearBounds
from .facade import BoundModel, Method
from .propagation import AlphaOptimizationConfig
from .regions import HyperRectangle

# A "method spec" is either a method name, or a (method, alpha_config) tuple
# requesting alpha-CROWN optimization for that method.
MethodSpec = Method | tuple[Method, AlphaOptimizationConfig]
PlotFn = Callable[..., torch.Tensor] | torch.nn.Module

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


__all__ = [
    "DEFAULT_METHODS",
    "BoundsGridEntry",
    "plot_bounds",
    "plot_bounds_grid",
]


DEFAULT_METHODS: tuple[Method, ...] = ("ibp", "forward_lbp", "backward_lbp")

_METHOD_COLORS: dict[Method, str] = {
    "ibp": "tab:red",
    "forward_lbp": "tab:blue",
    "backward_lbp": "tab:green",
    "forward_backward_lbp": "tab:purple",
    "crown_ibp": "tab:orange",
}

# Alpha-optimized variants reuse the base method's color but desaturated.
_ALPHA_COLORS: dict[Method, str] = {
    "ibp": "darkred",
    "forward_lbp": "navy",
    "backward_lbp": "darkgreen",
    "forward_backward_lbp": "indigo",
    "crown_ibp": "saddlebrown",
}


@dataclass(frozen=True)
class BoundsGridEntry:
    """One panel for :func:`plot_bounds_grid`: a function over a 1D region.

    Parameters
    ----------
    fn :
        The function or ``nn.Module`` to plot. Must accept a 1D tensor input
        of shape ``(1,)``.
    region :
        Input region with ``region.shape == (1,)``.
    title :
        Optional axes title. ``None`` leaves the panel untitled.
    """

    fn: PlotFn
    region: HyperRectangle
    title: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_bounds(
    fn: PlotFn,
    region: HyperRectangle,
    *,
    methods: Sequence[MethodSpec] = DEFAULT_METHODS,
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
    methods : sequence of method specs
        Each entry is either a method name (e.g. ``"backward_lbp"``) or a
        ``(method, AlphaOptimizationConfig)`` tuple to enable alpha-CROWN
        optimization for that method. Methods that fail on this op are
        reported in-axes and the others still render. Defaults to all three
        single-registry methods without alpha.
    num_samples : int
        Number of x-samples used to draw the true function curve.
    output_index : int
        For vector-valued ``fn``, which output component to plot
        (``f(x).flatten()[output_index]``).
    title : str, optional
        Axes title.
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
    _check_1d_region(region)

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
    for spec in methods:
        method, alpha_config = _normalize_method_spec(spec)
        label, color = _label_and_color(method, alpha_config)
        try:
            model = BoundModel(fn, dummy_inputs=(dummy,), method=method, alpha=alpha_config)
            bounds = model.propagate(region)
        except Exception as exc:  # noqa: BLE001
            _annotate_failure(ax, label, exc, color=color)
            continue
        _draw_bound(ax, bounds, xs, output_index=output_index, label=label, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    return fig  # ty:ignore[invalid-return-type]


def plot_bounds_grid(
    entries: Sequence[BoundsGridEntry],
    *,
    cols: int = 3,
    methods: Sequence[MethodSpec] = DEFAULT_METHODS,
    num_samples: int = 200,
    output_index: int = 0,
    panel_size: tuple[float, float] = (5.5, 3.5),
    suptitle: str | None = None,
) -> Figure:
    """Render one :func:`plot_bounds` panel per entry into a grid figure.

    Parameters
    ----------
    entries :
        Panels to render, in row-major order.
    cols :
        Number of columns in the grid. Rows are derived from ``len(entries)``.
        Clamped to ``[1, len(entries)]`` so a sparse last row is rare.
    methods :
        Method specs forwarded to every panel; see :func:`plot_bounds`.
    num_samples, output_index :
        Forwarded to every panel; see :func:`plot_bounds`.
    panel_size :
        ``(width, height)`` per panel in inches. Total figure size is
        ``(cols * width, rows * height)``.
    suptitle :
        Optional figure-level title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the grid. Empty trailing axes (if any) are
        hidden via ``ax.axis("off")``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``entries`` is empty or any entry's region is not 1-dimensional.
    """
    if not entries:
        raise ValueError("plot_bounds_grid requires at least one entry; got an empty sequence.")
    plt = _require_matplotlib()

    n = len(entries)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    width, height = panel_size
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    axes_flat = axes.flatten()

    for entry, ax in zip(entries, axes_flat, strict=False):
        plot_bounds(
            entry.fn,
            entry.region,
            methods=methods,
            num_samples=num_samples,
            output_index=output_index,
            title=entry.title,
            ax=ax,
        )

    for ax in axes_flat[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install matplotlib` "
            "or as an optional extra: `pip install bound_propagation[viz]`."
        ) from exc
    return plt


def _check_1d_region(region: HyperRectangle) -> None:
    if tuple(region.shape) != (1,):
        raise ValueError(f"plot_bounds requires a 1D input region (shape (1,)), got shape {tuple(region.shape)}")


def _normalize_method_spec(spec: MethodSpec) -> tuple[Method, AlphaOptimizationConfig | None]:
    if isinstance(spec, tuple):
        method, alpha_config = spec
        return method, alpha_config
    return spec, None


def _label_and_color(method: Method, alpha_config: AlphaOptimizationConfig | None) -> tuple[str, str | None]:
    if alpha_config is not None and alpha_config.enabled:
        return f"{method}+α", _ALPHA_COLORS.get(method, _METHOD_COLORS.get(method))
    return method, _METHOD_COLORS.get(method)


def _evaluate_function(fn: PlotFn, xs: torch.Tensor, output_index: int) -> torch.Tensor:
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
    """Render ``bounds`` onto ``ax``. Constant bounds → horizontal band; affine → two lines."""
    if isinstance(bounds, IntervalBounds):
        lower_val = float(bounds.lower.detach().flatten()[output_index])
        upper_val = float(bounds.upper.detach().flatten()[output_index])
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
        # 1D input region: input_size == 1, output_size == bias_lower.numel().
        output_size = bounds.bias_lower.numel()
        coeff = bounds.coefficient
        lower_dense = coeff.lower.to_dense().tensor.detach().reshape(output_size, -1)
        upper_dense = coeff.upper.to_dense().tensor.detach().reshape(output_size, -1)
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
