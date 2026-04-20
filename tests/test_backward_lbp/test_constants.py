"""Tests for treating torch.zeros/ones/full family as constants under backward LBP."""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, region


def test_zeros_literal_shape() -> None:
    def fn(x):
        return x + torch.zeros(3)

    r = region([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert_exact(fn, r, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0]))


def test_full_literal_shape() -> None:
    def fn(x):
        return x * torch.full((3,), 2.0)

    r = region([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert_exact(fn, r, torch.tensor([2.0, 2.0, 2.0]), torch.tensor([4.0, 4.0, 4.0]))


def test_zeros_like_abstract_input() -> None:
    def fn(x):
        return x + torch.zeros_like(x)

    r = region([1.0, 2.0], [2.0, 3.0])
    assert_exact(fn, r, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))


def test_ones_like_abstract_input() -> None:
    def fn(x):
        return x * torch.ones_like(x) + torch.ones_like(x)

    r = region([0.0, 1.0], [1.0, 2.0])
    assert_sound(fn, r)


def test_full_like_abstract_input() -> None:
    def fn(x):
        return x + torch.full_like(x, 5.0)

    r = region([0.0, 0.0], [1.0, 2.0])
    assert_exact(fn, r, torch.tensor([5.0, 5.0]), torch.tensor([6.0, 7.0]))
