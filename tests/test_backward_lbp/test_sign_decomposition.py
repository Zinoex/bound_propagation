"""Hand-computed sign-decomposition references for backward LBP.

Verifies that ``IntervalLeafRelaxation`` and ``ElementwiseBackwardRelaxation``
produce the analytic outputs for backward composition through the elementwise
relaxation pattern, with mixed-sign A matrices.

The reference formulas (auto_LiRPA equation 5, restricted to elementwise
``α·z + β`` relaxations) are:

    A_pos = A.clamp(min=0),  A_neg = A.clamp(max=0)
    new_A_lower = A_l_pos · α_lower + A_l_neg · α_upper
    new_A_upper = A_u_pos · α_upper + A_u_neg · α_lower
    Δb_lower    = (A_l_pos · β_lower + A_l_neg · β_upper).sum(node_dims)
    Δb_upper    = (A_u_pos · β_upper + A_u_neg · β_lower).sum(node_dims)

For the interval-leaf case the predecessor is concretized to ``[h_lo, h_up]``
and α / β collapse to ``α = 0, β_lower = h_lo, β_upper = h_up`` so the
contribution is purely to the bias.
"""

from __future__ import annotations

import torch
import torch.fx as fx

from bound_propagation.linear_operators import DenseOperator
from bound_propagation.propagation.backward_lbp.base import IntervalLeafRelaxation
from bound_propagation.propagation.backward_lbp.elementwise import ElementwiseBackwardRelaxation
from bound_propagation.propagation.linear_relaxations.elementwise import ElementwiseParams


def _placeholder_node(name: str = "x") -> fx.Node:
    """Construct a real ``fx.Node`` placeholder for use as a dict key."""
    graph = fx.Graph()
    return graph.placeholder(name)


def _signed_compose_reference(
    A_lower: torch.Tensor,
    A_upper: torch.Tensor,
    *,
    alpha_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_lower: torch.Tensor,
    beta_upper: torch.Tensor,
    node_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The analytic auto_LiRPA elementwise sign-decomposition formula."""
    A_l_pos, A_l_neg = A_lower.clamp(min=0), A_lower.clamp(max=0)
    A_u_pos, A_u_neg = A_upper.clamp(min=0), A_upper.clamp(max=0)

    new_A_lower = A_l_pos * alpha_lower + A_l_neg * alpha_upper
    new_A_upper = A_u_pos * alpha_upper + A_u_neg * alpha_lower
    delta_b_lower = (A_l_pos * beta_lower + A_l_neg * beta_upper).sum(dim=node_dims)
    delta_b_upper = (A_u_pos * beta_upper + A_u_neg * beta_lower).sum(dim=node_dims)
    return new_A_lower, new_A_upper, delta_b_lower, delta_b_upper


# ---------------------------------------------------------------------------
# IntervalLeafRelaxation
# ---------------------------------------------------------------------------


class TestIntervalLeafSignDecomposition:
    """Backward through an interval leaf: contribution is bias-only."""

    def test_positive_a_uses_lower_for_lower_upper_for_upper(self) -> None:
        # All-positive A: lower bound of A·h is A·h_lo, upper bound is A·h_up.
        h_lower = torch.tensor([1.0, -2.0, 0.5])
        h_upper = torch.tensor([2.0, -1.0, 1.5])
        # A shape: (output, node) = (2, 3); positive entries.
        A_tensor = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        A_lower = DenseOperator(A_tensor, output_shape=(2,))
        A_upper = DenseOperator(A_tensor, output_shape=(2,))

        relaxation = IntervalLeafRelaxation(lower=h_lower, upper=h_upper)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)  # ty:ignore[invalid-argument-type]

        assert contributions.a_terms == {}
        # Lower bound: positive A times h_lower.
        expected_lower = A_tensor @ h_lower
        # Upper bound: positive A times h_upper.
        expected_upper = A_tensor @ h_upper
        assert torch.allclose(contributions.bias_lower, expected_lower)
        assert torch.allclose(contributions.bias_upper, expected_upper)

    def test_negative_a_uses_upper_for_lower_lower_for_upper(self) -> None:
        # All-negative A: lower bound of A·h is A·h_up, upper bound is A·h_lo.
        h_lower = torch.tensor([1.0, 2.0, 3.0])
        h_upper = torch.tensor([2.0, 4.0, 6.0])
        A_tensor = torch.tensor([[-1.0, -2.0, -0.5]])
        A_lower = DenseOperator(A_tensor, output_shape=(1,))
        A_upper = DenseOperator(A_tensor, output_shape=(1,))

        relaxation = IntervalLeafRelaxation(lower=h_lower, upper=h_upper)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)  # ty:ignore[invalid-argument-type]

        expected_lower = A_tensor @ h_upper
        expected_upper = A_tensor @ h_lower
        assert torch.allclose(contributions.bias_lower, expected_lower)
        assert torch.allclose(contributions.bias_upper, expected_upper)

    def test_mixed_sign_a_matches_analytic_decomposition(self) -> None:
        h_lower = torch.tensor([1.0, -3.0, 2.0])
        h_upper = torch.tensor([4.0, -1.0, 5.0])
        # Hand-picked mixed signs to exercise both clamps in every output row.
        A_tensor = torch.tensor([[1.5, -2.0, 0.5], [-0.5, 1.0, -3.0]])
        A_lower = DenseOperator(A_tensor, output_shape=(2,))
        A_upper = DenseOperator(A_tensor, output_shape=(2,))

        relaxation = IntervalLeafRelaxation(lower=h_lower, upper=h_upper)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)  # ty:ignore[invalid-argument-type]

        A_pos = A_tensor.clamp(min=0)
        A_neg = A_tensor.clamp(max=0)
        expected_lower = A_pos @ h_lower + A_neg @ h_upper
        expected_upper = A_pos @ h_upper + A_neg @ h_lower
        assert torch.allclose(contributions.bias_lower, expected_lower)
        assert torch.allclose(contributions.bias_upper, expected_upper)

    def test_distinct_lower_upper_a_matrices(self) -> None:
        # When A_lower and A_upper differ, each is sign-decomposed independently.
        h_lower = torch.tensor([1.0, -2.0])
        h_upper = torch.tensor([3.0, 0.5])
        A_lower_t = torch.tensor([[1.0, -1.0]])
        A_upper_t = torch.tensor([[-2.0, 3.0]])
        A_lower = DenseOperator(A_lower_t, output_shape=(1,))
        A_upper = DenseOperator(A_upper_t, output_shape=(1,))

        relaxation = IntervalLeafRelaxation(lower=h_lower, upper=h_upper)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)  # ty:ignore[invalid-argument-type]

        Al_pos, Al_neg = A_lower_t.clamp(min=0), A_lower_t.clamp(max=0)
        Au_pos, Au_neg = A_upper_t.clamp(min=0), A_upper_t.clamp(max=0)
        expected_lower = Al_pos @ h_lower + Al_neg @ h_upper
        expected_upper = Au_pos @ h_upper + Au_neg @ h_lower
        assert torch.allclose(contributions.bias_lower, expected_lower)
        assert torch.allclose(contributions.bias_upper, expected_upper)


# ---------------------------------------------------------------------------
# ElementwiseBackwardRelaxation (sign decomposition against α·z + β relaxation)
# ---------------------------------------------------------------------------


class TestElementwiseSignDecomposition:
    """Backward through ``α·z + β``: A is decomposed into pos / neg parts."""

    def _params(self, *, alpha_lower, alpha_upper, beta_lower, beta_upper) -> ElementwiseParams:
        return ElementwiseParams(
            alpha_lower=torch.as_tensor(alpha_lower, dtype=torch.float32),
            alpha_upper=torch.as_tensor(alpha_upper, dtype=torch.float32),
            beta_lower=torch.as_tensor(beta_lower, dtype=torch.float32),
            beta_upper=torch.as_tensor(beta_upper, dtype=torch.float32),
        )

    def test_relu_like_pattern_matches_analytic_reference(self) -> None:
        # Simulate ReLU's relaxation on a 3-element node with a mix of regimes:
        # - element 0 (active): α = 1, β = 0
        # - element 1 (inactive): α = 0, β = 0
        # - element 2 (crossing): α_lower = 0, α_upper = 0.5, β_lower = 0, β_upper = 0.5
        alpha_lower = [1.0, 0.0, 0.0]
        alpha_upper = [1.0, 0.0, 0.5]
        beta_lower = [0.0, 0.0, 0.0]
        beta_upper = [0.0, 0.0, 0.5]
        params = self._params(
            alpha_lower=alpha_lower, alpha_upper=alpha_upper, beta_lower=beta_lower, beta_upper=beta_upper
        )

        # A shape: (bounded_out=2, node=3) with mixed signs.
        A_tensor = torch.tensor([[1.0, -1.0, 2.0], [-2.0, 0.5, 1.5]])
        A_lower = DenseOperator(A_tensor, output_shape=(2,))
        A_upper = DenseOperator(A_tensor.clone(), output_shape=(2,))

        node = _placeholder_node()
        relaxation = ElementwiseBackwardRelaxation(params=params, input_node=node)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)

        # Compare against the analytic formula. node_dims = (-1,) since node is the trailing axis.
        expected_A_lower, expected_A_upper, expected_db_lower, expected_db_upper = _signed_compose_reference(
            A_tensor,
            A_tensor,
            alpha_lower=params.alpha_lower,
            alpha_upper=params.alpha_upper,
            beta_lower=params.beta_lower,
            beta_upper=params.beta_upper,
            node_dims=(-1,),
        )
        # backward_through returns one a_term keyed by node.
        assert list(contributions.a_terms.keys()) == [node]
        new_A_lower_op, new_A_upper_op = contributions.a_terms[node]
        assert torch.allclose(new_A_lower_op.to_dense().tensor, expected_A_lower)
        assert torch.allclose(new_A_upper_op.to_dense().tensor, expected_A_upper)
        assert torch.allclose(contributions.bias_lower, expected_db_lower)
        assert torch.allclose(contributions.bias_upper, expected_db_upper)

    def test_distinct_lower_upper_a_matrices_swap_pairing(self) -> None:
        """Lower-side A pairs with α_lower (pos) / α_upper (neg); upper-side swaps."""
        alpha_lower = [0.5, 0.5, 0.0]
        alpha_upper = [1.0, 1.0, 0.5]
        beta_lower = [0.0, 0.0, 0.0]
        beta_upper = [0.0, 0.0, 0.5]
        params = self._params(
            alpha_lower=alpha_lower, alpha_upper=alpha_upper, beta_lower=beta_lower, beta_upper=beta_upper
        )

        A_lower_t = torch.tensor([[1.0, -2.0, 0.5]])
        A_upper_t = torch.tensor([[-1.0, 1.5, -0.5]])
        A_lower = DenseOperator(A_lower_t, output_shape=(1,))
        A_upper = DenseOperator(A_upper_t, output_shape=(1,))

        node = _placeholder_node()
        relaxation = ElementwiseBackwardRelaxation(params=params, input_node=node)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)

        expected_A_lower, expected_A_upper, expected_db_lower, expected_db_upper = _signed_compose_reference(
            A_lower_t,
            A_upper_t,
            alpha_lower=params.alpha_lower,
            alpha_upper=params.alpha_upper,
            beta_lower=params.beta_lower,
            beta_upper=params.beta_upper,
            node_dims=(-1,),
        )
        new_A_lower_op, new_A_upper_op = contributions.a_terms[node]
        assert torch.allclose(new_A_lower_op.to_dense().tensor, expected_A_lower)
        assert torch.allclose(new_A_upper_op.to_dense().tensor, expected_A_upper)
        assert torch.allclose(contributions.bias_lower, expected_db_lower)
        assert torch.allclose(contributions.bias_upper, expected_db_upper)

    def test_purely_affine_relaxation_zero_bias(self) -> None:
        """When α_lower == α_upper and β = 0, the relaxation is exact-affine."""
        alpha = [1.0, 1.0, 1.0]
        params = self._params(alpha_lower=alpha, alpha_upper=alpha, beta_lower=[0.0] * 3, beta_upper=[0.0] * 3)

        A_tensor = torch.tensor([[1.0, -1.0, 2.0]])
        A_lower = DenseOperator(A_tensor, output_shape=(1,))
        A_upper = DenseOperator(A_tensor.clone(), output_shape=(1,))

        node = _placeholder_node()
        relaxation = ElementwiseBackwardRelaxation(params=params, input_node=node)
        contributions = relaxation.backward_through(A_lower, A_upper, 0)

        new_A_lower_op, new_A_upper_op = contributions.a_terms[node]
        # A * α with α = 1 should leave A unchanged on both sides.
        assert torch.allclose(new_A_lower_op.to_dense().tensor, A_tensor)
        assert torch.allclose(new_A_upper_op.to_dense().tensor, A_tensor)
        assert torch.allclose(contributions.bias_lower, torch.zeros(1))
        assert torch.allclose(contributions.bias_upper, torch.zeros(1))
