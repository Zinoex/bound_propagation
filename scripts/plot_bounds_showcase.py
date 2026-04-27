"""Sanity-check showcase: plot IBP / forward-LBP / backward-LBP bounds against
the true function for a curated set of operations and small compositions.

Usage
-----
    uv run python scripts/plot_bounds_showcase.py
    uv run python scripts/plot_bounds_showcase.py --output bounds_showcase.png
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from bound_propagation import HyperRectangle
from bound_propagation.visualization import BoundsGridEntry, plot_bounds_grid


def _hyperrect_1d(lo: float, hi: float) -> HyperRectangle:
    return HyperRectangle(lower=torch.tensor([lo]), upper=torch.tensor([hi]))


def _entry(title: str, fn, lo: float, hi: float) -> BoundsGridEntry:
    return BoundsGridEntry(fn=fn, region=_hyperrect_1d(lo, hi), title=title)


def _affine_layer(seed: int, in_features: int = 1, out_features: int = 1) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features)


def _two_layer_relu_mlp(seed: int) -> nn.Module:
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


def _four_layer_relu_mlp(seed: int, *, hidden: int = 8) -> nn.Module:
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


def _relu_sigmoid_classifier_head(seed: int) -> nn.Module:
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


# Curated entries — region picked to keep ops well-defined and visibly nonlinear.
ENTRIES: list[BoundsGridEntry] = [
    # Elementwise nonlinearities.
    _entry("relu", torch.relu, -2.0, 2.0),
    _entry("sigmoid", torch.sigmoid, -4.0, 4.0),
    _entry("tanh", torch.tanh, -3.0, 3.0),
    _entry("exp", torch.exp, -1.5, 1.5),
    _entry("log", torch.log, 0.1, 4.0),
    _entry("sqrt", torch.sqrt, 0.05, 4.0),
    _entry("sin", torch.sin, -math.pi, math.pi),
    _entry("cos", torch.cos, -math.pi, math.pi),
    _entry("abs", torch.abs, -2.0, 2.0),
    _entry("reciprocal", torch.reciprocal, 0.2, 4.0),
    _entry("clamp(-1, 1)", lambda x: torch.clamp(x, -1.0, 1.0), -2.0, 2.0),
    # Composition: x*x via multiplication.
    _entry("x * x", lambda x: x * x, -1.5, 1.5),
    # Affine layers + small MLPs.
    _entry("Linear(1→1)", _affine_layer(seed=0), -1.0, 1.0),
    _entry("Linear → Sigmoid", _linear_then_sigmoid(seed=1), -2.0, 2.0),
    _entry("Linear → ReLU → Linear", _two_layer_relu_mlp(seed=2), -2.0, 2.0),
    # Function composition.
    _entry("sigmoid(2x + 1)", lambda x: torch.sigmoid(2 * x + 1), -3.0, 3.0),
    _entry("exp(-x*x)", lambda x: torch.exp(-(x * x)), -2.0, 2.0),
    _entry("tanh(x) - relu(x)", lambda x: torch.tanh(x) - torch.relu(x), -2.0, 2.0),
    # Deeper compositions — three Linear layers with two activation stages.
    _entry("Linear→ReLU→Linear→ReLU→Linear", _three_layer_mlp(seed=10, activation=nn.ReLU), -1.5, 1.5),
    _entry("Linear→Tanh→Linear→Tanh→Linear", _three_layer_mlp(seed=11, activation=nn.Tanh), -1.5, 1.5),
    _entry("Mixed (Sigmoid + Tanh)", _mixed_activation_mlp(seed=12), -2.0, 2.0),
    _entry("Deep MLP: 4×Linear, 3×ReLU", _four_layer_relu_mlp(seed=13), -1.0, 1.0),
    _entry("MLP head with Sigmoid", _relu_sigmoid_classifier_head(seed=14), -1.5, 1.5),
    # Hand-built nonlinear composition.
    _entry("tanh(sin(x) * x)", lambda x: torch.tanh(torch.sin(x) * x), -math.pi, math.pi),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="bounds_showcase.png", help="Output PNG path (default: %(default)s)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples per panel")
    args = parser.parse_args()

    fig = plot_bounds_grid(ENTRIES, cols=args.cols, num_samples=args.num_samples)
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output} ({len(ENTRIES)} panels)")


if __name__ == "__main__":
    main()
