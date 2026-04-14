from __future__ import annotations

from itertools import product

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.matmul import IBPMatmul

from tests.helpers import propagate


def _normalize_like_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
    if left.ndim == 0 or right.ndim == 0:
        raise ValueError("matmul requires tensor inputs with at least 1 dimension")

    squeeze_left = left.ndim == 1
    squeeze_right = right.ndim == 1

    if squeeze_left:
        left = left.unsqueeze(0)
    if squeeze_right:
        right = right.unsqueeze(-1)

    if left.shape[-1] != right.shape[-2]:
        raise ValueError(
            "matmul requires compatible reduction dimensions, "
            f"got left.shape={tuple(left.shape)} and right.shape={tuple(right.shape)}"
        )

    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(*batch_shape, left.shape[-2], left.shape[-1])
    right = right.expand(*batch_shape, right.shape[-2], right.shape[-1])

    return left, right, squeeze_left, squeeze_right


def _restore_like_matmul(result: torch.Tensor, squeeze_left: bool, squeeze_right: bool) -> torch.Tensor:
    if squeeze_left:
        result = result.squeeze(-2)
    if squeeze_right:
        result = result.squeeze(-1)
    return result


def _bruteforce_interval_interval_matmul(left: IntervalBounds, right: IntervalBounds) -> IntervalBounds:
    left_lower, right_lower, squeeze_left, squeeze_right = _normalize_like_matmul(left.lower, right.lower)
    left_upper, _, _, _ = _normalize_like_matmul(left.upper, right.lower)
    _, right_upper, _, _ = _normalize_like_matmul(left.lower, right.upper)

    output_shape = left_lower.shape[:-1] + (right_lower.shape[-1],)
    lower = torch.empty(output_shape, dtype=left_lower.dtype, device=left_lower.device)
    upper = torch.empty(output_shape, dtype=left_lower.dtype, device=left_lower.device)

    batch_shape = left_lower.shape[:-2]
    m = left_lower.shape[-2]
    k = left_lower.shape[-1]
    n = right_lower.shape[-1]

    for batch_index in product(*(range(dim) for dim in batch_shape)) if batch_shape else [()]:
        for i in range(m):
            for j in range(n):
                min_value = 0.0
                max_value = 0.0

                for t in range(k):
                    a_lower = left_lower[*batch_index, i, t]
                    b_lower = right_lower[*batch_index, t, j]

                    a_upper = left_upper[*batch_index, i, t]
                    b_upper = right_upper[*batch_index, t, j]

                    ll = float(a_lower * b_lower)
                    lu = float(a_lower * b_upper)
                    ul = float(a_upper * b_lower)
                    uu = float(a_upper * b_upper)

                    min_value += min(ll, lu, ul, uu)
                    max_value += max(ll, lu, ul, uu)

                lower[*batch_index, i, j] = min_value
                upper[*batch_index, i, j] = max_value

    return IntervalBounds(
        _restore_like_matmul(lower, squeeze_left, squeeze_right),
        _restore_like_matmul(upper, squeeze_left, squeeze_right),
    )


def _assert_matches_bruteforce(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> None:
    strategy = IBPMatmul()
    left = IntervalBounds(left_lower, left_upper)
    right = IntervalBounds(right_lower, right_upper)

    actual = propagate(strategy, left, right)
    expected = _bruteforce_interval_interval_matmul(left, right)

    assert torch.allclose(actual.lower, expected.lower, atol=1e-6)
    assert torch.allclose(actual.upper, expected.upper, atol=1e-6)


def test_interval_interval_square_negative_intervals() -> None:
    _assert_matches_bruteforce(
        left_lower=torch.tensor([[-4.0, -3.0], [-2.0, -5.0]]),
        left_upper=torch.tensor([[-2.0, -1.0], [-1.0, -3.0]]),
        right_lower=torch.tensor([[-6.0, -2.0], [-3.0, -4.0]]),
        right_upper=torch.tensor([[-4.0, -1.0], [-1.0, -2.0]]),
    )


def test_interval_interval_wide_positive_intervals() -> None:
    _assert_matches_bruteforce(
        left_lower=torch.tensor([[1.0], [2.0]]),
        left_upper=torch.tensor([[3.0], [4.0]]),
        right_lower=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        right_upper=torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
    )


def test_interval_interval_long_mixed_intervals() -> None:
    _assert_matches_bruteforce(
        left_lower=torch.tensor([[-1.5, -0.2, 0.1, 1.0]]),
        left_upper=torch.tensor([[0.5, 0.4, 1.2, 2.0]]),
        right_lower=torch.tensor([[-2.0, 0.0], [1.0, -1.0], [-0.5, 2.0], [0.1, -3.0]]),
        right_upper=torch.tensor([[-1.0, 1.0], [2.0, 1.0], [0.5, 3.0], [1.0, -1.0]]),
    )


def test_interval_interval_vector_cases_include_zero() -> None:
    # Vector @ matrix -> vector
    _assert_matches_bruteforce(
        left_lower=torch.tensor([-1.0, 0.0, 2.0]),
        left_upper=torch.tensor([1.0, 2.0, 3.0]),
        right_lower=torch.tensor([[1.0, -2.0], [-1.0, 0.0], [2.0, -1.0]]),
        right_upper=torch.tensor([[2.0, -1.0], [1.0, 1.0], [3.0, 2.0]]),
    )

    # Matrix @ vector -> vector
    _assert_matches_bruteforce(
        left_lower=torch.tensor([[-2.0, 0.5, 1.0], [0.0, -1.0, 2.0]]),
        left_upper=torch.tensor([[-1.0, 1.5, 2.0], [1.0, 0.0, 3.0]]),
        right_lower=torch.tensor([-1.0, 0.0, -2.0]),
        right_upper=torch.tensor([1.0, 2.0, -1.0]),
    )

    # Vector @ vector -> scalar
    _assert_matches_bruteforce(
        left_lower=torch.tensor([-1.0, 0.0, 2.0]),
        left_upper=torch.tensor([1.0, 1.0, 3.0]),
        right_lower=torch.tensor([-2.0, -1.0, 0.5]),
        right_upper=torch.tensor([-1.0, 2.0, 1.5]),
    )


def test_interval_interval_batched_broadcasted_shapes() -> None:
    _assert_matches_bruteforce(
        left_lower=torch.tensor(
            [
                [[-1.0, 0.0]],
            ]
        ),
        left_upper=torch.tensor(
            [
                [[1.0, 2.0]],
            ]
        ),
        right_lower=torch.tensor(
            [
                [[-2.0], [1.0]],
                [[0.5], [-1.5]],
                [[2.0], [0.0]],
            ]
        ),
        right_upper=torch.tensor(
            [
                [[-1.0], [2.0]],
                [[1.5], [-0.5]],
                [[3.0], [1.0]],
            ]
        ),
    )


def test_interval_interval_nontrivial_batch_broadcast_t1_with_u() -> None:
    # Left batch dims (t, 1), right batch dims (u,). Output batch dims must be (t, u).
    # Shapes: (t, 1, m, k) @ (u, k, n) -> (t, u, m, n)
    _assert_matches_bruteforce(
        left_lower=torch.tensor(
            [
                [
                    [[-1.0, 0.0], [1.0, -2.0]],
                ],
                [
                    [[0.5, -1.5], [2.0, 1.0]],
                ],
            ]
        ),
        left_upper=torch.tensor(
            [
                [
                    [[2.0, 1.0], [2.5, -0.5]],
                ],
                [
                    [[1.5, -0.5], [3.0, 2.0]],
                ],
            ]
        ),
        right_lower=torch.tensor(
            [
                [[-2.0, 1.0, 0.5], [1.0, -1.0, -0.5]],
                [[0.0, -1.0, 2.0], [0.5, 1.0, -2.0]],
                [[-1.0, 2.0, -1.5], [2.0, -0.5, 0.0]],
            ]
        ),
        right_upper=torch.tensor(
            [
                [[-1.0, 2.0, 1.5], [2.0, 0.0, 0.5]],
                [[1.0, 0.0, 3.0], [1.5, 2.0, -1.0]],
                [[0.0, 3.0, -0.5], [3.0, 0.5, 1.0]],
            ]
        ),
    )


def test_interval_interval_batched_matrix_vector_branch() -> None:
    # Shapes: (b1, b2, m, k) @ (k,) -> (b1, b2, m)
    _assert_matches_bruteforce(
        left_lower=torch.tensor(
            [
                [
                    [[-1.0, 0.5, 2.0], [1.0, -2.0, 0.0]],
                    [[0.0, 1.0, -1.0], [2.0, -0.5, 0.5]],
                ]
            ]
        ),
        left_upper=torch.tensor(
            [
                [
                    [[0.5, 1.5, 3.0], [2.0, -1.0, 1.0]],
                    [[1.0, 2.0, 0.0], [3.0, 0.5, 1.5]],
                ]
            ]
        ),
        right_lower=torch.tensor([-2.0, 0.0, 1.0]),
        right_upper=torch.tensor([-1.0, 1.0, 2.0]),
    )


def test_interval_interval_batched_vector_matrix_branch() -> None:
    # Shapes: (b, k) @ (1, k, n) -> (b, n) via batch broadcast.
    _assert_matches_bruteforce(
        left_lower=torch.tensor(
            [
                [-1.0, 0.0, 2.0],
                [0.5, -2.0, 1.0],
                [1.0, 1.5, -0.5],
            ]
        ),
        left_upper=torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.5, -1.0, 2.0],
                [2.0, 2.5, 0.5],
            ]
        ),
        right_lower=torch.tensor(
            [
                [[-2.0, 1.0], [0.5, -1.0], [1.0, 0.0]],
            ]
        ),
        right_upper=torch.tensor(
            [
                [[-1.0, 2.0], [1.5, 0.0], [2.0, 1.0]],
            ]
        ),
    )
