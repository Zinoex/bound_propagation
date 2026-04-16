"""Tests for constant-over-bounds division linear relaxation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from bound_propagation.propagation.linear_relaxations.constant_div import compute_constant_div_relaxation


def _eval_line(alpha: torch.Tensor, beta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return alpha * x + beta


def test_constant_div_positive_nominal_sound() -> None:
    lower = torch.tensor([2.0])
    upper = torch.tensor([4.0])

    relaxation = compute_constant_div_relaxation(lower, upper, constant=6.0)
    alpha_lower = relaxation.alpha_lower
    beta_lower = relaxation.beta_lower
    alpha_upper = relaxation.alpha_upper
    beta_upper = relaxation.beta_upper

    x = torch.linspace(2.0, 4.0, 200)
    y_true = 6.0 / x
    y_lower = _eval_line(alpha_lower, beta_lower, x)
    y_upper = _eval_line(alpha_upper, beta_upper, x)

    assert torch.all(y_lower <= y_true + 1e-5)
    assert torch.all(y_upper >= y_true - 1e-5)


def test_constant_div_negative_nominal_sound() -> None:
    lower = torch.tensor([2.0])
    upper = torch.tensor([4.0])

    relaxation = compute_constant_div_relaxation(lower, upper, constant=-6.0)
    alpha_lower = relaxation.alpha_lower
    beta_lower = relaxation.beta_lower
    alpha_upper = relaxation.alpha_upper
    beta_upper = relaxation.beta_upper

    x = torch.linspace(2.0, 4.0, 200)
    y_true = -6.0 / x
    y_lower = _eval_line(alpha_lower, beta_lower, x)
    y_upper = _eval_line(alpha_upper, beta_upper, x)

    assert torch.all(y_lower <= y_true + 1e-5)
    assert torch.all(y_upper >= y_true - 1e-5)


def test_constant_div_crossing_zero_returns_infinite_bounds() -> None:
    lower = torch.tensor([-1.0, 2.0])
    upper = torch.tensor([1.0, 3.0])

    relaxation = compute_constant_div_relaxation(lower, upper, constant=6.0)
    alpha_lower = relaxation.alpha_lower
    beta_lower = relaxation.beta_lower
    alpha_upper = relaxation.alpha_upper
    beta_upper = relaxation.beta_upper

    assert torch.isneginf(beta_lower[0])
    assert torch.isposinf(beta_upper[0])
    assert torch.isfinite(beta_lower[1])
    assert torch.isfinite(beta_upper[1])
    assert torch.all(alpha_lower[0:1] == 0)
    assert torch.all(alpha_upper[0:1] == 0)


def test_constant_div_zero_constant_is_exact_zero() -> None:
    lower = torch.tensor([-1.0, 2.0])
    upper = torch.tensor([1.0, 3.0])

    relaxation = compute_constant_div_relaxation(lower, upper, constant=0.0)
    alpha_lower = relaxation.alpha_lower
    beta_lower = relaxation.beta_lower
    alpha_upper = relaxation.alpha_upper
    beta_upper = relaxation.beta_upper

    assert torch.all(alpha_lower == 0)
    assert torch.all(beta_lower == 0)
    assert torch.all(alpha_upper == 0)
    assert torch.all(beta_upper == 0)


def test_constant_div_broadcast_constant_tensor() -> None:
    lower = torch.tensor([2.0, 2.0])
    upper = torch.tensor([4.0, 4.0])
    constant = torch.tensor([6.0, -6.0])

    relaxation = compute_constant_div_relaxation(lower, upper, constant=constant)
    alpha_lower = relaxation.alpha_lower
    beta_lower = relaxation.beta_lower
    alpha_upper = relaxation.alpha_upper
    beta_upper = relaxation.beta_upper

    x = torch.tensor([3.0, 3.0])
    y_true = constant / x
    y_lower = _eval_line(alpha_lower, beta_lower, x)
    y_upper = _eval_line(alpha_upper, beta_upper, x)

    assert torch.all(y_lower <= y_true + 1e-5)
    assert torch.all(y_upper >= y_true - 1e-5)
