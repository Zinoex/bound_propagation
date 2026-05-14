"""Sanity check for the compatibility test harness."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ._harness import ALL_METHODS, check_op_compatibility, make_region


@pytest.mark.parametrize("method", ALL_METHODS)
def test_harness_runs_linear(method: str) -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 4)
    region = make_region((3,), lower=-0.5, upper=0.5)
    check_op_compatibility(model, (torch.zeros(3),), [region], method)
