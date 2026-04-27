"""Tests for the optional matplotlib-based ``plot_bounds`` helper.

Skipped when matplotlib is not installed (the soft dependency).
"""

from __future__ import annotations

import pytest
import torch

# The module is import-safe even without matplotlib (lazy import inside the
# plotting function); the actual plotting calls require it.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # noqa: E402 — must precede any pyplot use, even indirectly.

from bound_propagation import HyperRectangle  # noqa: E402
from bound_propagation.visualization import (  # noqa: E402
    DEFAULT_METHODS,
    BoundsGridEntry,
    plot_bounds,
    plot_bounds_grid,
)


def _region(lo: float, hi: float) -> HyperRectangle:
    return HyperRectangle(lower=torch.tensor([lo]), upper=torch.tensor([hi]))


class TestPlotBoundsBasic:
    def test_relu_returns_figure(self) -> None:
        fig = plot_bounds(lambda x: torch.relu(x), _region(-1.0, 1.0), num_samples=50)
        assert fig is not None
        # Sanity: figure has at least one axes; the axes contains plotted lines / patches.
        assert len(fig.axes) >= 1
        ax = fig.axes[0]
        # The true function plus per-method artists should populate something.
        assert len(ax.lines) + len(ax.collections) > 0

    def test_methods_appear_in_legend(self) -> None:
        fig = plot_bounds(lambda x: torch.tanh(x), _region(-1.0, 1.0), num_samples=20)
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        # The true function and at least one method label per requested method.
        assert "f(x)" in labels
        for method in DEFAULT_METHODS:
            assert any(method in label for label in labels), (
                f"Expected a legend entry mentioning {method!r}; got {labels}"
            )

    def test_subset_of_methods(self) -> None:
        fig = plot_bounds(lambda x: torch.sigmoid(x), _region(-2.0, 2.0), methods=("ibp",), num_samples=20)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert any("ibp" in label for label in labels)
        assert not any("forward_lbp" in label for label in labels)

    def test_method_failure_does_not_crash(self) -> None:
        # An op the propagator can't trace — this should produce a labeled
        # failure entry rather than raising.
        def unsupported(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([float(x.item()) ** 0.5])  # uses .item() (untraceable in fx)

        fig = plot_bounds(unsupported, _region(0.5, 2.0), num_samples=10)
        # The plot still rendered the true function.
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("f(x)" in label for label in labels)

    def test_rejects_non_1d_region(self) -> None:
        region = HyperRectangle(lower=torch.zeros(2), upper=torch.ones(2))
        with pytest.raises(ValueError, match="1D input region"):
            plot_bounds(lambda x: x.sum().reshape(1), region)

    def test_renders_on_supplied_axes(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned = plot_bounds(lambda x: torch.relu(x), _region(-1.0, 1.0), ax=ax, num_samples=20)
        assert returned is fig

    def test_vector_output_uses_output_index(self) -> None:
        # A two-output function: pick the second component.
        torch.manual_seed(0)
        layer = torch.nn.Linear(1, 2)
        fig = plot_bounds(layer, _region(-1.0, 1.0), output_index=1, num_samples=20)
        # Just need it to render without error.
        assert fig is not None


class TestPlotBoundsGrid:
    def test_renders_one_panel_per_entry(self) -> None:
        entries = [
            BoundsGridEntry(fn=torch.relu, region=_region(-1.0, 1.0), title="relu"),
            BoundsGridEntry(fn=torch.tanh, region=_region(-2.0, 2.0), title="tanh"),
            BoundsGridEntry(fn=torch.sigmoid, region=_region(-3.0, 3.0), title="sigmoid"),
        ]
        fig = plot_bounds_grid(entries, cols=2, num_samples=20)
        # rows = ceil(3 / 2) = 2, total axes = rows * cols = 4 (one hidden).
        assert len(fig.axes) == 4
        active_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert active_titles == ["relu", "tanh", "sigmoid"]

    def test_hides_trailing_axes_when_grid_underfilled(self) -> None:
        entries = [BoundsGridEntry(fn=torch.relu, region=_region(-1.0, 1.0), title="relu")]
        fig = plot_bounds_grid(entries, cols=3, num_samples=10)
        # cols clamps to len(entries) so no underfill — exactly one axes.
        assert len(fig.axes) == 1

    def test_clamps_cols_to_entry_count(self) -> None:
        entries = [
            BoundsGridEntry(fn=torch.relu, region=_region(-1.0, 1.0)),
            BoundsGridEntry(fn=torch.tanh, region=_region(-1.0, 1.0)),
        ]
        fig = plot_bounds_grid(entries, cols=10, num_samples=10)
        # cols clamped to 2 → 1 row.
        assert len(fig.axes) == 2

    def test_suptitle_is_set(self) -> None:
        entries = [BoundsGridEntry(fn=torch.relu, region=_region(-1.0, 1.0))]
        fig = plot_bounds_grid(entries, suptitle="my title", num_samples=10)
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "my title"

    def test_rejects_empty_entries(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            plot_bounds_grid([])

    def test_propagates_methods_argument(self) -> None:
        entries = [BoundsGridEntry(fn=torch.relu, region=_region(-1.0, 1.0), title="relu")]
        fig = plot_bounds_grid(entries, methods=("ibp",), num_samples=10)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert any("ibp" in label for label in labels)
        assert not any("forward_lbp" in label for label in labels)
