"""Sanity-check showcase #2: reduction (sum, mean, amax, amin) and shape
(flatten, reshape, cat, stack, transpose, permute, view, select, getitem,
unsqueeze, squeeze) operations.

Each entry takes a scalar (shape ``(1,)``) input, internally constructs the
multi-dim tensor needed by the showcased op, and returns a scalar so the same
``plot_bounds`` helper can render it. This keeps the visualization 1D while
still exercising every reduction and shape strategy end-to-end.

Usage
-----
    uv run python scripts/plot_bounds_showcase_shapes.py
    uv run python scripts/plot_bounds_showcase_shapes.py --output my_shapes.png
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from bound_propagation import HyperRectangle
from bound_propagation.visualization import plot_bounds


@dataclass(frozen=True)
class ShowcaseEntry:
    title: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    region_lower: float
    region_upper: float


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _sum_three_offsets(x: torch.Tensor) -> torch.Tensor:
    """``sum([x, x+1, x-1]) = 3x`` — affine, bounds should be tight."""
    return torch.stack([x, x + 1.0, x - 1.0]).sum()


def _mean_three_scaled(x: torch.Tensor) -> torch.Tensor:
    """``mean([x, 2x, 3x]) = 2x`` — affine."""
    return torch.stack([x, x * 2.0, x * 3.0]).mean()


def _amax_paths(x: torch.Tensor) -> torch.Tensor:
    """``amax([x, -x, x*x - 1])`` — chain-breaking interval leaf in backward LBP."""
    return torch.amax(torch.stack([x, -x, x * x - 1.0]))


def _amin_paths(x: torch.Tensor) -> torch.Tensor:
    """``amin([sin(x), cos(x), x*0.5])``."""
    return torch.amin(torch.stack([torch.sin(x), torch.cos(x), x * 0.5]))


def _sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """``sum([x^2, (x+1)^2]) = 2x^2 + 2x + 1`` — quadratic via stack + mul + sum."""
    return (torch.stack([x, x + 1.0]) ** 2).sum()


def _mean_of_relus(x: torch.Tensor) -> torch.Tensor:
    """``mean(relu([x-1, x, x+1]))`` — sum of ReLU pieces normalized; piecewise linear."""
    return torch.relu(torch.stack([x - 1.0, x, x + 1.0])).mean()


def _sum_of_sigmoids(x: torch.Tensor) -> torch.Tensor:
    """``sum(sigmoid([x, x+2, x-2]))`` — three-sigmoid sum (smooth nonlinear)."""
    return torch.sigmoid(torch.stack([x, x + 2.0, x - 2.0])).sum()


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------


def _flatten_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``cat([x]*6).reshape(2,3).flatten().sum() = 6x`` — exercises ``flatten``."""
    return torch.stack([x, x, x, x, x, x]).reshape(2, 3).flatten().sum()


def _reshape_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``stack([x]*4).reshape(2,2).sum() = 4x`` — exercises ``reshape``."""
    return torch.stack([x, x, x, x]).reshape(2, 2).sum()


def _view_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``stack([x]*6).view(2,3).sum() = 6x`` — exercises ``view``."""
    return torch.stack([x, x, x, x, x, x]).view(2, 3).sum()


def _cat_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``cat([x, 2x, 3x]).sum() = 6x`` — exercises ``cat``."""
    return torch.cat([x, x * 2.0, x * 3.0]).sum()


def _stack_then_amax(x: torch.Tensor) -> torch.Tensor:
    """``amax(stack([x, sin(x), cos(x), -x*0.5]))`` — exercises ``stack``."""
    return torch.amax(torch.stack([x, torch.sin(x), torch.cos(x), -x * 0.5]))


def _transpose_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``stack([stack([x]*3)]*2).transpose(0,1).sum() = 6x`` — exercises ``transpose``."""
    row = torch.stack([x, x, x])  # shape (3, 1)
    return torch.stack([row, row]).transpose(0, 1).sum()


def _permute_then_sum(x: torch.Tensor) -> torch.Tensor:
    """A nested-stack permute that contracts to ``24x`` — exercises ``permute``."""
    inner = torch.stack([x, x, x, x])  # (4, 1)
    middle = torch.stack([inner, inner, inner])  # (3, 4, 1)
    block = torch.stack([middle, middle])  # (2, 3, 4, 1)
    return block.permute(3, 0, 1, 2).sum()


def _select_middle(x: torch.Tensor) -> torch.Tensor:
    """``stack([x, 2x, 3x]).select(0, 1).reshape(1) = 2x`` — exercises ``select``."""
    return torch.stack([x, x * 2.0, x * 3.0]).select(0, 1).reshape(1)


def _getitem_index(x: torch.Tensor) -> torch.Tensor:
    """``stack([x, sin(x), cos(x)])[1].reshape(1) = sin(x)`` — exercises ``__getitem__``."""
    return torch.stack([x, torch.sin(x), torch.cos(x)])[1].reshape(1)


def _unsqueeze_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``unsqueeze(stack([x, x, x]), 0).sum() = 3x`` — exercises ``unsqueeze``."""
    return torch.stack([x, x, x]).unsqueeze(0).sum()


def _squeeze_then_relu(x: torch.Tensor) -> torch.Tensor:
    """``relu(stack([x, -x]).squeeze()).sum() = relu(x) + relu(-x) = |x|``."""
    return torch.relu(torch.stack([x, -x]).squeeze(-1)).sum()


# ---------------------------------------------------------------------------
# Mixed reduction + shape
# ---------------------------------------------------------------------------


def _mean_of_path(x: torch.Tensor) -> torch.Tensor:
    """``mean([x, sin(x), cos(x)])`` — three-component reduction over a nonlinear stack."""
    return torch.stack([x, torch.sin(x), torch.cos(x)]).mean()


def _amax_minus_amin(x: torch.Tensor) -> torch.Tensor:
    """Range over a nonlinear stack: ``amax(s) - amin(s)`` for ``s = [x, sin(x), cos(x), x/2]``."""
    s = torch.stack([x, torch.sin(x), torch.cos(x), x * 0.5])
    return torch.amax(s) - torch.amin(s)


def _flatten_then_relu_then_sum(x: torch.Tensor) -> torch.Tensor:
    """``relu([x-1, x, x+1].flatten()).sum()`` — chained shape + nonlinear + reduction."""
    return torch.relu(torch.stack([x - 1.0, x, x + 1.0]).flatten()).sum()


# ---------------------------------------------------------------------------
# Curated entries (region picked to keep ops well-defined)
# ---------------------------------------------------------------------------

ENTRIES: list[ShowcaseEntry] = [
    # Reductions.
    ShowcaseEntry("sum([x, x+1, x-1]) = 3x", _sum_three_offsets, -2.0, 2.0),
    ShowcaseEntry("mean([x, 2x, 3x]) = 2x", _mean_three_scaled, -2.0, 2.0),
    ShowcaseEntry("amax([x, -x, x²-1])", _amax_paths, -2.0, 2.0),
    ShowcaseEntry("amin([sin x, cos x, x/2])", _amin_paths, -math.pi, math.pi),
    ShowcaseEntry("sum([x², (x+1)²])", _sum_of_squares, -2.0, 2.0),
    ShowcaseEntry("mean(relu([x-1, x, x+1]))", _mean_of_relus, -2.0, 2.0),
    ShowcaseEntry("sum(sigmoid([x, x+2, x-2]))", _sum_of_sigmoids, -3.0, 3.0),
    # Shape ops.
    ShowcaseEntry("flatten → sum = 6x", _flatten_then_sum, -1.5, 1.5),
    ShowcaseEntry("reshape → sum = 4x", _reshape_then_sum, -1.5, 1.5),
    ShowcaseEntry("view → sum = 6x", _view_then_sum, -1.5, 1.5),
    ShowcaseEntry("cat → sum = 6x", _cat_then_sum, -1.5, 1.5),
    ShowcaseEntry("amax(stack([x, sin x, cos x, -x/2]))", _stack_then_amax, -math.pi, math.pi),
    ShowcaseEntry("transpose → sum = 6x", _transpose_then_sum, -1.5, 1.5),
    ShowcaseEntry("permute → sum = 24x", _permute_then_sum, -0.5, 0.5),
    ShowcaseEntry("select(stack, 0, 1) = 2x", _select_middle, -1.5, 1.5),
    ShowcaseEntry("getitem(stack, 1) = sin x", _getitem_index, -math.pi, math.pi),
    ShowcaseEntry("unsqueeze → sum = 3x", _unsqueeze_then_sum, -1.5, 1.5),
    ShowcaseEntry("squeeze + relu = |x|", _squeeze_then_relu, -2.0, 2.0),
    # Mixed.
    ShowcaseEntry("mean([x, sin x, cos x])", _mean_of_path, -math.pi, math.pi),
    ShowcaseEntry("amax(s) - amin(s)", _amax_minus_amin, -math.pi, math.pi),
    ShowcaseEntry("relu([x-1, x, x+1]).sum()", _flatten_then_relu_then_sum, -2.0, 2.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="bounds_showcase_shapes.png",
        help="Output PNG path (default: %(default)s)",
    )
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per panel")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "This showcase requires matplotlib. Install with `pip install matplotlib` or `uv sync --group dev`."
        ) from exc

    n_entries = len(ENTRIES)
    cols = args.cols
    rows = (n_entries + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for entry, ax in zip(ENTRIES, axes_flat, strict=False):
        region = HyperRectangle(
            lower=torch.tensor([entry.region_lower]),
            upper=torch.tensor([entry.region_upper]),
        )
        plot_bounds(entry.fn, region, num_samples=args.num_samples, title=entry.title, ax=ax)

    for ax in axes_flat[n_entries:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output} ({n_entries} panels)")


if __name__ == "__main__":
    main()
