from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.tan import IBPTan

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    strategy = IBPTan()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_tan_no_asymptote_interval_stays_finite() -> None:
    out = _propagate(
        lower=torch.tensor([0.0], dtype=torch.float64),
        upper=torch.tensor([0.4], dtype=torch.float64),
    )

    assert torch.isfinite(out.lower).all()
    assert torch.isfinite(out.upper).all()
    assert torch.allclose(out.lower, torch.tan(torch.tensor([0.0], dtype=torch.float64)))
    assert torch.allclose(out.upper, torch.tan(torch.tensor([0.4], dtype=torch.float64)))


def test_tan_asymptote_at_lower_endpoint_is_detected() -> None:
    pi_over_2 = torch.pi / 2
    out = _propagate(
        lower=torch.tensor([pi_over_2], dtype=torch.float64),
        upper=torch.tensor([pi_over_2 + 0.1], dtype=torch.float64),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_tan_asymptote_at_upper_endpoint_is_detected() -> None:
    pi_over_2 = torch.pi / 2
    out = _propagate(
        lower=torch.tensor([pi_over_2 - 0.1], dtype=torch.float64),
        upper=torch.tensor([pi_over_2], dtype=torch.float64),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_tan_asymptote_inside_interval_is_detected() -> None:
    out = _propagate(
        lower=torch.tensor([1.4], dtype=torch.float64),
        upper=torch.tensor([1.8], dtype=torch.float64),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_tan_batched_mixed_intervals() -> None:
    pi_over_2 = torch.pi / 2
    out = _propagate(
        lower=torch.tensor([0.0, pi_over_2, -1.0, -10.0], dtype=torch.float64),
        upper=torch.tensor([0.2, pi_over_2 + 0.2, 0.5, 5.0], dtype=torch.float64),
    )

    # no asymptote
    assert torch.isfinite(out.lower[0])
    assert torch.isfinite(out.upper[0])

    # lower endpoint asymptote
    assert torch.isneginf(out.lower[1])
    assert torch.isposinf(out.upper[1])

    # no asymptote
    assert torch.isfinite(out.lower[2])
    assert torch.isfinite(out.upper[2])

    # multiple asymptotes
    assert torch.isneginf(out.lower[3])
    assert torch.isposinf(out.upper[3])
