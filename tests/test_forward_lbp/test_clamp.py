from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.clamp import ForwardLBPClamp
from bound_propagation.regions import HyperRectangle


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        region=region,
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_clamp_interval_within_range() -> None:
    """Test clamp when interval is already within clamp range."""
    # Region: x ∈ [3, 5]
    # clamp(x, min=0, max=10) = x (since 3 >= 0 and 5 <= 10)
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should be identity (alpha=1, beta=0)
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_clamp_interval_below_range() -> None:
    """Test clamp when interval is completely below clamp range."""
    # Region: x ∈ [-5, -2]
    # clamp(x, min=0, max=10) = 0 (since -5 < 0 and -2 < 0)
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-2.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should be constant at 0
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([0.0]))


def test_clamp_interval_above_range() -> None:
    """Test clamp when interval is completely above clamp range."""
    # Region: x ∈ [12, 15]
    # clamp(x, min=0, max=10) = 10 (since 12 > 10)
    region = HyperRectangle(lower=torch.tensor([12.0]), upper=torch.tensor([15.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should be constant at 10
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([10.0]))
    assert torch.allclose(upper, torch.tensor([10.0]))


def test_clamp_crosses_min() -> None:
    """Test clamp when interval crosses the minimum threshold."""
    # Region: x ∈ [-2, 5]
    # clamp(x, min=0, max=10) = [0, 5]
    # Lower bound: line through (0, 0) with slope = (5-0)/(5-(-2)) = 5/7
    # Upper bound: line connecting (-2, 0) and (5, 5) with slope = 5/7
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # At x=-2: clamp(-2, 0, 10) = 0
    # At x=5: clamp(5, 0, 10) = 5
    assert torch.all(lower >= -0.1)
    assert torch.all(lower <= 0.1)
    assert torch.all(upper >= 4.9)
    assert torch.all(upper <= 5.1)


def test_clamp_crosses_max() -> None:
    """Test clamp when interval crosses the maximum threshold."""
    # Region: x ∈ [3, 12]
    # clamp(x, min=0, max=10) = [3, 10]
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([12.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # At x=3: clamp(3, 0, 10) = 3
    # At x=12: clamp(12, 0, 10) = 10
    assert torch.all(lower >= 2.9)
    assert torch.all(lower <= 3.1)
    assert torch.all(upper >= 9.9)
    assert torch.all(upper <= 10.1)


def test_clamp_crosses_both() -> None:
    """Test clamp when interval crosses both min and max thresholds."""
    # Region: x ∈ [-5, 15]
    # clamp(x, min=0, max=10) = [0, 10]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([15.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should be [0, 10] with linear relaxations
    assert torch.all(lower >= -0.1)
    assert torch.all(lower <= 0.5)
    assert torch.all(upper >= 9.5)
    assert torch.all(upper <= 10.1)


def test_clamp_only_min() -> None:
    """Test clamp with only min specified."""
    # Region: x ∈ [-2, 5]
    # clamp(x, min=0) = [0, 5]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": None}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= -0.1)
    assert torch.all(lower <= 0.1)
    assert torch.all(upper >= 4.9)
    assert torch.all(upper <= 5.1)


def test_clamp_only_max() -> None:
    """Test clamp with only max specified."""
    # Region: x ∈ [3, 12]
    # clamp(x, max=10) = [3, 10]
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([12.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": None, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= 2.9)
    assert torch.all(lower <= 3.1)
    assert torch.all(upper >= 9.9)
    assert torch.all(upper <= 10.1)


def test_clamp_point_interval() -> None:
    """Test clamp on a point interval."""
    # Region: x ∈ [5, 5]
    # clamp(x, min=0, max=10) = 5
    region = HyperRectangle(lower=torch.tensor([5.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    class MockNode:
        def __init__(self):
            self.attributes = {"min": 0.0, "max": 10.0}

    node = MockNode()
    strategy = ForwardLBPClamp()
    result = strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([5.0]), atol=1e-6)
