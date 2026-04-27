"""Sanity-check showcase #3: a single deeply-nonlinear composition rendered
across every supported propagation method.

The function under test is a small MLP-like composition with three different
activations stacked through several affine layers, plus a hand-built
nonlinear coupling on top so multiplications and elementwise functions all
participate. This stresses the deep-network looseness gap between IBP and
LBP, and between standalone LBP and alpha-CROWN-optimized LBP.

Methods rendered (one figure):

- ``ibp``                                 — pure interval bound propagation
- ``forward_lbp``                         — forward-mode CROWN-style relaxation
- ``backward_lbp``                        — standard CROWN
- ``forward_backward_lbp``                — forward intermediate bounds + backward final
- ``crown_ibp``                           — IBP intermediate bounds + backward final
- ``forward_lbp`` + α                     — α-CROWN on the forward LBP pass
- ``backward_lbp`` + α                    — α-CROWN on the standard backward pass
- ``forward_backward_lbp`` + α            — α-CROWN on the backward final pass
- ``crown_ibp`` + α                       — α-CROWN on the backward final pass

Every LBP variant supports α-optimization on its backward path (only IBP
doesn't). The four LBP+α variants tend to converge to nearly the same width
on smooth functions; the un-optimized variants can differ by more.

Usage
-----
    uv run python scripts/plot_bounds_showcase_complex.py
    uv run python scripts/plot_bounds_showcase_complex.py --output complex.png \\
        --alpha-iters 50 --grid 2
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from bound_propagation import HyperRectangle
from bound_propagation.propagation import AlphaOptimizationConfig
from bound_propagation.visualization import BoundsGridEntry, plot_bounds_grid


class ComplexComposition(nn.Module):
    """Function under test, packaged as one ``nn.Module`` so the inner MLP is
    fx-traceable as a submodule (closure-captured modules can't be traced —
    ``fx`` resolves submodule paths by attribute lookup).

    Computation:

    - Inner MLP: ``Linear → ReLU → Linear → Sigmoid → Linear → Tanh → Linear``
      (mix of saturating + piecewise-linear activations across four affine stages).
    - Top-level composition: ``g(x) = sin(x)*sigmoid(x) + 0.3*net(x) + tanh(x*x)``,
      exercising mul, sin, sigmoid, the deep MLP, another mul, tanh, and add.
    """

    def __init__(self, hidden: int = 8, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(1, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, 1)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(x))
        h = torch.sigmoid(self.l2(h))
        h = torch.tanh(self.l3(h))
        return self.l4(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x) * torch.sigmoid(x)
        b = 0.3 * self._net(x).squeeze(-1)  # squeezed to scalar to exercise broadcast add
        c = torch.tanh(x * x)
        return (a + b + c).reshape(1)


def _build_methods(alpha_cfg: AlphaOptimizationConfig) -> tuple:
    """Every supported propagation mode + α-CROWN variants for each LBP-based one.

    α-optimization is available on the backward path of every LBP variant —
    including the dual-registry methods (``forward_backward_lbp``, ``crown_ibp``),
    which use backward LBP for the final pass. IBP has no α knobs.
    """
    base_lbp_methods = ("forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp")
    return ("ibp", *base_lbp_methods, *((m, alpha_cfg) for m in base_lbp_methods))


def _build_grid_entries(fn: nn.Module, region_lo: float, region_hi: float, n_regions: int) -> list[BoundsGridEntry]:
    """Split ``[region_lo, region_hi]`` into ``n_regions`` equal sub-regions, one panel each."""
    edges = torch.linspace(region_lo, region_hi, n_regions + 1)
    entries: list[BoundsGridEntry] = []
    for i in range(n_regions):
        lo, hi = edges[i].reshape(1), edges[i + 1].reshape(1)
        title = f"complex(x), x ∈ [{float(lo):.2f}, {float(hi):.2f}]"
        entries.append(BoundsGridEntry(fn=fn, region=HyperRectangle(lower=lo, upper=hi), title=title))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="bounds_showcase_complex.png",
        help="Output PNG path (default: %(default)s)",
    )
    parser.add_argument("--num-samples", type=int, default=300, help="x-samples for the true curve")
    parser.add_argument(
        "--alpha-iters",
        type=int,
        default=30,
        help="alpha-CROWN optimization iterations for the alpha variant",
    )
    parser.add_argument("--alpha-lr", type=float, default=0.1, help="alpha-CROWN learning rate")
    parser.add_argument(
        "--grid",
        type=int,
        default=1,
        help="Number of regions along the x axis. >1 splits the panel into a grid of panels, "
        "each over a sub-region. Useful for narrow regions where alpha-CROWN really shines.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Network init seed")
    parser.add_argument("--region-lo", type=float, default=-math.pi / 2)
    parser.add_argument("--region-hi", type=float, default=math.pi / 2)
    args = parser.parse_args()

    fn = ComplexComposition(seed=args.seed).eval()
    alpha_cfg = AlphaOptimizationConfig(
        enabled=True,
        iterations=args.alpha_iters,
        lr=args.alpha_lr,
        loss="width",
        optimize_intermediate=False,
    )
    methods = _build_methods(alpha_cfg)
    entries = _build_grid_entries(fn, args.region_lo, args.region_hi, max(1, args.grid))

    fig = plot_bounds_grid(
        entries,
        cols=2,
        methods=methods,
        num_samples=args.num_samples,
        panel_size=(7.5, 5.0),
        suptitle=(
            f"Deeply-nonlinear composition — {len(methods)} methods compared "
            f"(α: {args.alpha_iters} iters, lr={args.alpha_lr})"
        ),
    )
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output} ({len(entries)} panel(s), {len(methods)} methods each)")


if __name__ == "__main__":
    main()
