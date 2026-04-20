"""Full-workflow tests for patch-mode forward-LBP (``Conv2dPatchOperator``).

Verifies that a ``conv → relu → conv`` (and deeper) chain stays structural
via :class:`Conv2dPatchOperator`, that bounds are sound, and that the
structured bounds match a dense reference numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle
from bound_propagation.linear_operators import Conv2dPatchOperator, DenseOperator
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardLBPPropagator
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(model: nn.Module, dummy: torch.Tensor):
    registry = create_default_forward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(model)
    MetadataPass(gm).run(dummy)
    return gm


def _make_region(shape: tuple[int, ...], width: float = 0.3, seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + width
    return HyperRectangle(lower=lower, upper=upper)


def _check_sound(model, region: HyperRectangle, bounds, n: int = 50) -> None:
    concrete = bounds.concretize()
    for _ in range(n):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = model(x)
        assert torch.all(concrete.lower <= y + 1e-5), (concrete.lower, y)
        assert torch.all(y <= concrete.upper + 1e-5), (concrete.upper, y)


class TestPatchModeForConvReluConv:
    def test_conv_relu_conv_emits_patch_operator(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 4, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4))
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])

        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_uppers_op)
        _check_sound(model, region, out)

    def test_conv_relu_conv_matches_dense_reference(self) -> None:
        """Patch-mode concretization must match materialized dense concretize."""
        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=1)
        gm = _trace_and_annotate(model, region.lower)
        out_patch = ForwardLBPPropagator(gm).propagate([region])
        lo_p, up_p = out_patch.concretize()

        # Dense reference: materialize each patch op, re-build LinearBounds.
        from bound_propagation.bounds import LinearBounds  # noqa: PLC0415

        dense_lower = [op.to_dense() for op in out_patch.linear_lowers_op]
        dense_upper = [op.to_dense() for op in out_patch.linear_uppers_op]
        dense_bounds = LinearBounds(
            regions=out_patch.regions,
            linear_lower=dense_lower,
            bias_lower=out_patch.bias_lower,
            linear_upper=dense_upper,
            bias_upper=out_patch.bias_upper,
            input_ids=out_patch.input_ids,
        )
        lo_d, up_d = dense_bounds.concretize()
        assert torch.allclose(lo_p, lo_d, atol=1e-4)
        assert torch.allclose(up_p, up_d, atol=1e-4)


class TestPatchModeForDeeperChains:
    def test_conv_relu_conv_relu(self) -> None:
        """conv₁ → relu → conv₂ → relu stays structured (the last ReLU scales
        the Conv2dPatchOperator, which falls back to dense for now; still sound)."""
        torch.manual_seed(2)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.c2(torch.relu(self.c1(x))))

        model = Net()
        region = _make_region((2, 4, 4), seed=2)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])
        # After the second ReLU, patches survive because Conv2dPatchOperator.scale
        # stays structural (multiplies the patch tensor by the per-output alpha).
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_lowers_op)
        _check_sound(model, region, out)

    def test_conv_relu_conv_with_different_channels(self) -> None:
        torch.manual_seed(3)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(3, 5, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(5, 7, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((3, 5, 5), seed=3)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_lowers_op)
        _check_sound(model, region, out)


class TestPatchModeDenseFallback:
    def test_strided_second_conv_falls_back(self) -> None:
        """A strided second conv doesn't compose structurally → dense fallback."""
        torch.manual_seed(4)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=2, stride=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=4)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])
        # Stride=2 on the second conv triggers the fallback to DenseOperator.
        assert all(isinstance(op, DenseOperator) for op in out.linear_lowers_op)
        _check_sound(model, region, out)


class TestPatchModeFacadeCNN:
    def test_cnn_with_patch_mode_chain(self) -> None:
        """Full facade: conv→relu→conv chain embedded in a larger CNN."""
        torch.manual_seed(5)
        model = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        dummy = torch.zeros(2, 8, 8)
        region = _make_region((2, 8, 8), width=0.2, seed=5)
        bm = BoundModel(model, dummy_inputs=(dummy,), method="forward_lbp")
        bounds = bm.propagate(region)
        lo, up = bounds.concretize()
        assert lo.shape == (4, 4, 4)
        for _ in range(30):
            x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
            y = model(x)
            assert torch.all(lo <= y + 1e-4)
            assert torch.all(y <= up + 1e-4)
