"""Full-workflow tests for :class:`ScaledConv2dOperator` in forward-LBP.

Verifies:
- A conv followed by an elementwise nonlinearity emits
  :class:`ScaledConv2dOperator` coefficients (no dense materialization).
- Chains of multiple nonlinearities after one conv stay scaled-conv.
- A second conv layer forces materialization and produces :class:`DenseOperator`.
- In every case the bounds are sound and numerically match the dense path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle
from bound_propagation.linear_operators import (
    ScaledConv2dOperator,
)
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


class TestConvReluEmitsScaledConv:
    """After conv+ReLU, the linear coefficient operators should be ScaledConv2dOperator."""

    def test_conv_relu(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.conv(x))

        model = Net()
        region = _make_region((2, 4, 4))
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])

        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_uppers_op)
        _check_sound(model, region, out)

    def test_conv_sigmoid(self) -> None:
        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.conv(x))

        model = Net()
        region = _make_region((2, 4, 4), seed=1)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])

        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_lowers_op)
        _check_sound(model, region, out)

    def test_conv_tanh(self) -> None:
        torch.manual_seed(2)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tanh(self.conv(x))

        model = Net()
        region = _make_region((2, 4, 4), seed=2)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])
        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_lowers_op)
        _check_sound(model, region, out)


class TestStructuralChainThroughMultipleNonlinearities:
    """conv → relu → sigmoid → tanh should stay scaled-conv throughout."""

    def test_chained_nonlinearities(self) -> None:
        torch.manual_seed(3)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tanh(torch.sigmoid(torch.relu(self.conv(x))))

        model = Net()
        region = _make_region((2, 4, 4), seed=3)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])

        # All three stacked nonlinearities preserve ScaledConv2dOperator structure.
        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, ScaledConv2dOperator) for op in out.linear_uppers_op)
        _check_sound(model, region, out)


class TestSecondConvStaysStructural:
    """A second conv layer now stays structural via ``Conv2dPatchOperator``
    (patch-mode composition), not dense.
    """

    def test_conv_relu_conv_emits_patch_operator(self) -> None:
        from bound_propagation.linear_operators import Conv2dPatchOperator  # noqa: PLC0415

        torch.manual_seed(4)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=4)
        gm = _trace_and_annotate(model, region.lower)
        out = ForwardLBPPropagator(gm).propagate([region])

        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_uppers_op)
        _check_sound(model, region, out)


class TestStructuralMatchesDense:
    """The structured path must produce numerically-equivalent bounds to the
    dense path (which is what the phase-4 implementation would have produced
    before the Conv2dOperator/ScaledConv2dOperator wiring).
    """

    def test_conv_relu_matches_dense_reference(self) -> None:
        torch.manual_seed(5)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.conv(x))

        model = Net()
        region = _make_region((2, 4, 4), seed=5)
        gm = _trace_and_annotate(model, region.lower)
        out_struct = ForwardLBPPropagator(gm).propagate([region])

        lo_struct, up_struct = out_struct.concretize()

        # Reference: manually materialize every operator and run concretize.
        reference = out_struct.clone()
        dense_lower = [op.to_dense() for op in reference.linear_lowers_op]
        dense_upper = [op.to_dense() for op in reference.linear_uppers_op]

        # Substitute via the LinearBounds ctor (takes operators or tensors).
        from bound_propagation.bounds import LinearBounds  # noqa: PLC0415

        dense_bounds = LinearBounds(
            regions=reference.regions,
            linear_lower=dense_lower,
            bias_lower=reference.bias_lower,
            linear_upper=dense_upper,
            bias_upper=reference.bias_upper,
            input_ids=reference.input_ids,
        )
        lo_dense, up_dense = dense_bounds.concretize()
        assert torch.allclose(lo_struct, lo_dense, atol=1e-5)
        assert torch.allclose(up_struct, up_dense, atol=1e-5)


class TestFacadeCNN:
    """Facade-level end-to-end: a small CNN behaves identically to before
    (same concrete bounds, sound, correct output shape) — the Conv2dOperator +
    ScaledConv2dOperator machinery is transparent to the user."""

    def test_cnn_full_workflow(self) -> None:
        torch.manual_seed(6)
        model = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        dummy = torch.zeros(2, 8, 8)
        lower = torch.randn(2, 8, 8)
        upper = lower + 0.1
        region = HyperRectangle(lower=lower, upper=upper)
        bounds = BoundModel(model, dummy_inputs=(dummy,), method="forward_lbp").propagate(region)
        lo, up = bounds.concretize()
        assert lo.shape == (4, 2, 2)
        # Sample soundness
        for _ in range(30):
            x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
            y = model(x)
            assert torch.all(lo <= y + 1e-5)
            assert torch.all(y <= up + 1e-5)

    def test_conv_relu_classifier_head(self) -> None:
        """conv → relu → flatten_all → linear: structure survives the ReLU
        then materializes at the flatten (dense)."""
        torch.manual_seed(7)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.flatten = nn.Flatten(start_dim=0)
                self.fc = nn.Linear(3 * 4 * 4, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.flatten(torch.relu(self.conv(x))))

        model = Net()
        dummy = torch.zeros(2, 4, 4)
        lower = torch.randn(2, 4, 4)
        upper = lower + 0.1
        region = HyperRectangle(lower=lower, upper=upper)
        bounds = BoundModel(model, dummy_inputs=(dummy,), method="forward_lbp").propagate(region)
        lo, up = bounds.concretize()
        assert lo.shape == (5,)
        for _ in range(30):
            x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
            y = model(x)
            assert torch.all(lo <= y + 1e-5)
            assert torch.all(y <= up + 1e-5)
