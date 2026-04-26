"""Sanity-check showcase: plot IBP / forward-LBP / backward-LBP bounds against
the true function for a curated set of operations and small compositions.

Usage
-----
    uv run python scripts/plot_bounds_showcase.py
    uv run python scripts/plot_bounds_showcase.py --output bounds_showcase.png

By default writes ``bounds_showcase.png`` in the working directory.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from bound_propagation import HyperRectangle
from bound_propagation.visualization import plot_bounds


@dataclass(frozen=True)
class ShowcaseEntry:
    title: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    region_lower: float
    region_upper: float


def _affine_layer(in_features: int = 1, out_features: int = 1, *, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    layer = nn.Linear(in_features, out_features)
    return layer


def _two_layer_mlp(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))


def _linear_then_sigmoid(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())


def _three_layer_mlp(seed: int, *, hidden: int = 8, activation: type[nn.Module] = nn.ReLU) -> nn.Module:
    """``Linear → act → Linear → act → Linear`` — the canonical deep nonlinear case."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(1, hidden),
        activation(),
        nn.Linear(hidden, hidden),
        activation(),
        nn.Linear(hidden, 1),
    )


def _four_layer_mlp(seed: int, *, hidden: int = 8) -> nn.Module:
    """``(Linear → ReLU)×3 → Linear`` — bounds visibly degrade with depth (especially IBP)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def _mixed_activation_mlp(seed: int) -> nn.Module:
    """``Linear → Sigmoid → Linear → Tanh → Linear`` — heterogeneous nonlinearities."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(1, 6),
        nn.Sigmoid(),
        nn.Linear(6, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
    )


def _relu_sigmoid_head(seed: int) -> nn.Module:
    """``Linear → ReLU → Linear → ReLU → Linear → Sigmoid`` — classification-style head."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(1, 6),
        nn.ReLU(),
        nn.Linear(6, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
        nn.Sigmoid(),
    )


# Curated entries — each picks a region that respects the op's domain.
ENTRIES: list[ShowcaseEntry] = [
    # Elementwise nonlinearities.
    ShowcaseEntry("relu", lambda x: torch.relu(x), -2.0, 2.0),
    ShowcaseEntry("sigmoid", lambda x: torch.sigmoid(x), -4.0, 4.0),
    ShowcaseEntry("tanh", lambda x: torch.tanh(x), -3.0, 3.0),
    ShowcaseEntry("exp", lambda x: torch.exp(x), -1.5, 1.5),
    ShowcaseEntry("log", lambda x: torch.log(x), 0.1, 4.0),
    ShowcaseEntry("sqrt", lambda x: torch.sqrt(x), 0.05, 4.0),
    ShowcaseEntry("sin", lambda x: torch.sin(x), -math.pi, math.pi),
    ShowcaseEntry("cos", lambda x: torch.cos(x), -math.pi, math.pi),
    ShowcaseEntry("abs", lambda x: torch.abs(x), -2.0, 2.0),
    ShowcaseEntry("reciprocal", lambda x: torch.reciprocal(x), 0.2, 4.0),
    ShowcaseEntry("clamp(-1, 1)", lambda x: torch.clamp(x, -1.0, 1.0), -2.0, 2.0),
    # Composition: x*x via multiplication.
    ShowcaseEntry("x * x", lambda x: x * x, -1.5, 1.5),
    # Affine layers + small MLPs.
    ShowcaseEntry("Linear(1→1)", _affine_layer(seed=0), -1.0, 1.0),
    ShowcaseEntry("Linear → Sigmoid", _linear_then_sigmoid(seed=1), -2.0, 2.0),
    ShowcaseEntry("Linear → ReLU → Linear", _two_layer_mlp(seed=2), -2.0, 2.0),
    # Function composition: sigmoid(2x + 1).
    ShowcaseEntry("sigmoid(2x + 1)", lambda x: torch.sigmoid(2 * x + 1), -3.0, 3.0),
    # exp(-x^2) — gaussian-like, exercises mul + exp.
    ShowcaseEntry("exp(-x*x)", lambda x: torch.exp(-(x * x)), -2.0, 2.0),
    # tanh(x) - relu(x) — composition with subtraction.
    ShowcaseEntry("tanh(x) - relu(x)", lambda x: torch.tanh(x) - torch.relu(x), -2.0, 2.0),
    # Deeper compositions — three Linear layers with two ReLU stages
    # (canonical 2-hidden-layer MLP).
    ShowcaseEntry(
        "Linear→ReLU→Linear→ReLU→Linear",
        _three_layer_mlp(seed=10, hidden=8, activation=nn.ReLU),
        -1.5,
        1.5,
    ),
    ShowcaseEntry(
        "Linear→Tanh→Linear→Tanh→Linear",
        _three_layer_mlp(seed=11, hidden=8, activation=nn.Tanh),
        -1.5,
        1.5,
    ),
    ShowcaseEntry(
        "Mixed (Sigmoid + Tanh)",
        _mixed_activation_mlp(seed=12),
        -2.0,
        2.0,
    ),
    ShowcaseEntry(
        "Deep MLP: 4×Linear, 3×ReLU",
        _four_layer_mlp(seed=13, hidden=8),
        -1.0,
        1.0,
    ),
    ShowcaseEntry(
        "MLP head with Sigmoid",
        _relu_sigmoid_head(seed=14),
        -1.5,
        1.5,
    ),
    # Hand-built nonlinear composition: tanh(sin(x) * x).
    ShowcaseEntry(
        "tanh(sin(x) * x)",
        lambda x: torch.tanh(torch.sin(x) * x),
        -math.pi,
        math.pi,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="bounds_showcase.png", help="Output PNG path (default: %(default)s)")
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

    # Hide leftover axes if the grid is bigger than the entries list.
    for ax in axes_flat[n_entries:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output} ({n_entries} panels)")


if __name__ == "__main__":
    main()
