"""Backward LBP strategies and relaxations for linear / affine operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

from ...bounds import IntervalBounds
from ..linear_relaxations.alpha_resolvers import resolve_matmul_etas
from ..linear_relaxations.pairwise import PairedParams, compute_mul_relaxation
from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    accumulate_a_terms,
)

if TYPE_CHECKING:
    from .tape import BackwardTape


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class BackwardLBPLinear(BackwardLBPStrategy):
    """Backward LBP strategy for ``nn.Linear`` / ``F.linear``."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, kwargs = tape.resolve_args(node)

        if node.op == "call_module":
            module = tape.get_module(node.target)
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

        if weight is None:
            raise ValueError("BackwardLBPLinear requires a weight tensor")

        if weight.ndim not in (1, 2):
            raise ValueError(f"linear weight must be 1D or 2D, got shape {tuple(weight.shape)}")  # ty:ignore[invalid-argument-type]

        return LinearBackwardRelaxation(weight=weight, bias=bias, input_node=node.args[0])


class BackwardLBPMatmul(BackwardLBPStrategy):
    """Backward LBP strategy for matmul (abstract @ constant or constant @ abstract)."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, BackwardRelaxation) and isinstance(right, torch.Tensor):
            return MatmulRightConstantRelaxation(weight=right, input_node=node.args[0])

        if isinstance(left, torch.Tensor) and isinstance(right, BackwardRelaxation):
            return MatmulLeftConstantRelaxation(weight=left, input_node=node.args[1])

        if isinstance(left, BackwardRelaxation) and isinstance(right, BackwardRelaxation):
            left_node: fx.Node = node.args[0]  # ty:ignore[invalid-assignment]
            right_node: fx.Node = node.args[1]  # ty:ignore[invalid-assignment]
            bounds_a = bounds(left_node)
            bounds_b = bounds(right_node)

            if bounds_a.lower.ndim < 2 or bounds_b.lower.ndim < 2:
                raise NotImplementedError(
                    "Backward LBP matmul with two abstract operands requires each operand "
                    f"to be at least 2D (matrix), got shapes {tuple(bounds_a.lower.shape)} "
                    f"and {tuple(bounds_b.lower.shape)}."
                )

            params = _build_matmul_mccormick_params(bounds_a, bounds_b, node, tape)
            return MatmulBothAbstractRelaxation(
                params=params,
                left_node=left_node,
                right_node=right_node,
            )

        raise TypeError(
            f"BackwardLBPMatmul requires at least one BackwardRelaxation operand, got {type(left)} and {type(right)}"
        )


def _build_matmul_mccormick_params(
    bounds_a: IntervalBounds,
    bounds_b: IntervalBounds,
    node: fx.Node,
    tape: BackwardTape,
) -> PairedParams:
    """Build McCormick ``PairedParams`` for ``z = a @ b``.

    Returns params with element-wise shape ``(*batch, M, K, N)``. See
    :func:`bound_propagation.propagation.forward_lbp.matmul.ForwardLBPMatmul._matmul_bounds_bounds`
    for the derivation of the shape conventions.
    """
    try:
        batch_shape = torch.broadcast_shapes(
            bounds_a.lower.shape[:-2],
            bounds_b.lower.shape[:-2],
        )
    except RuntimeError as error:
        raise ValueError(
            "matmul requires broadcastable batch dimensions, "
            f"got a.shape={tuple(bounds_a.lower.shape)} and b.shape={tuple(bounds_b.lower.shape)}"
        ) from error

    m_dim = bounds_a.lower.shape[-2]
    k_a = bounds_a.lower.shape[-1]
    k_b = bounds_b.lower.shape[-2]
    n_dim = bounds_b.lower.shape[-1]

    if k_a != k_b:
        raise ValueError(f"matmul reduction dims mismatch: a.shape[-1]={k_a} vs b.shape[-2]={k_b}")

    la = bounds_a.lower.expand(*batch_shape, m_dim, k_a).unsqueeze(-1)
    ua = bounds_a.upper.expand(*batch_shape, m_dim, k_a).unsqueeze(-1)
    lb = bounds_b.lower.expand(*batch_shape, k_a, n_dim).unsqueeze(-3)
    ub = bounds_b.upper.expand(*batch_shape, k_a, n_dim).unsqueeze(-3)

    reference = IntervalBounds(
        la.expand(*batch_shape, m_dim, k_a, n_dim),
        ua.expand(*batch_shape, m_dim, k_a, n_dim),
    )
    eta_lo, eta_up = resolve_matmul_etas(tape.alpha_provider, node, reference)

    params = compute_mul_relaxation(
        IntervalBounds(la, ua),
        IntervalBounds(lb, ub),
        eta_lower=eta_lo if eta_lo is not None else 0.5,
        eta_upper=eta_up if eta_up is not None else 0.5,
    )

    # Broadcast each param to the full (*batch, M, K, N) grid so the
    # backward-pass reshapes land on a consistent 3-D per-node feature space.
    target = (*batch_shape, m_dim, k_a, n_dim)
    return PairedParams(
        alpha_lower_a=params.alpha_lower_a.expand(target).contiguous(),
        alpha_upper_a=params.alpha_upper_a.expand(target).contiguous(),
        alpha_lower_b=params.alpha_lower_b.expand(target).contiguous(),
        alpha_upper_b=params.alpha_upper_b.expand(target).contiguous(),
        bias_lower=params.bias_lower.expand(target).contiguous(),
        bias_upper=params.bias_upper.expand(target).contiguous(),
    )


class BackwardLBPAdd(BackwardLBPStrategy):
    """Backward LBP strategy for addition."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, BackwardRelaxation) and isinstance(right, BackwardRelaxation):
            return AddRelaxation(
                left_node=node.args[0],
                right_node=node.args[1],
                input_ndim=len(tape.shape_of(node.args[0])),
            )

        if isinstance(left, BackwardRelaxation):
            constant = torch.as_tensor(right, dtype=tape.dtype_of(node.args[0]))
            return ConstantAddRelaxation(constant=constant, input_node=node.args[0])

        if isinstance(right, BackwardRelaxation):
            constant = torch.as_tensor(left, dtype=tape.dtype_of(node.args[1]))
            return ConstantAddRelaxation(constant=constant, input_node=node.args[1])

        raise TypeError(
            f"BackwardLBPAdd requires at least one BackwardRelaxation operand, got {type(left)} and {type(right)}"
        )


class BackwardLBPSub(BackwardLBPStrategy):
    """Backward LBP strategy for subtraction."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, BackwardRelaxation) and isinstance(right, BackwardRelaxation):
            return SubRelaxation(
                left_node=node.args[0],
                right_node=node.args[1],
                input_ndim=len(tape.shape_of(node.args[0])),
            )

        if isinstance(left, BackwardRelaxation):
            # x - c = x + (-c)
            constant = torch.as_tensor(right, dtype=tape.dtype_of(node.args[0]))
            return ConstantAddRelaxation(constant=-constant, input_node=node.args[0])

        if isinstance(right, BackwardRelaxation):
            # c - x: bias gets +c contribution, A gets negated for x
            constant = torch.as_tensor(left, dtype=tape.dtype_of(node.args[1]))
            return ConstantAddRelaxation(constant=constant, input_node=node.args[1], negate_input=True)

        raise TypeError(
            f"BackwardLBPSub requires at least one BackwardRelaxation operand, got {type(left)} and {type(right)}"
        )


class BackwardLBPNeg(BackwardLBPStrategy):
    """Backward LBP strategy for negation."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        return NegRelaxation(
            input_node=node.args[0],
            input_ndim=len(tape.shape_of(node.args[0])),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _zero_bias(A: torch.Tensor, node_ndim: int) -> torch.Tensor:
    """Create a zero bias tensor with shape ``(*batch, *bounded_out)``.

    Parameters
    ----------
    A : torch.Tensor
        An A-matrix with shape ``(*batch, *bounded_out, *node)``.
    node_ndim : int
        Number of trailing dimensions that belong to the node output.
    """
    bias_shape = A.shape[: A.ndim - node_ndim] if node_ndim > 0 else A.shape
    return torch.zeros(bias_shape, dtype=A.dtype, device=A.device)


def _node_feature_ndim(node_ndim: int, batch_ndim: int) -> int:
    """Number of non-batch feature dimensions for a relaxation whose output
    has ``node_ndim`` total dimensions (captured at build time)."""
    return max(node_ndim - batch_ndim, 0)


# ---------------------------------------------------------------------------
# Relaxation dataclasses
# ---------------------------------------------------------------------------


@final
@dataclass
class LinearBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for ``nn.Linear`` / ``F.linear``: ``y = x @ W^T + b``.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor. Either 2D of shape ``(out_features, in_features)`` — in
        which case the output gains a trailing feature dim — or 1D of shape
        ``(in_features,)``, in which case the input's last dim is reduced via
        dot product and the output has no trailing feature dim.
    bias : torch.Tensor | None
        Bias tensor of shape ``(out_features,)`` for a 2D weight, or a scalar
        (0-D) tensor for a 1D weight. ``None`` if no bias.
    input_node : fx.Node
        The fx graph node for the input to this linear layer.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        if self.weight.ndim == 2:
            # A: (*batch, *bounded_out, out_features)
            # weight: (out_features, in_features) -> A @ weight gives (..., in_features)
            new_A_lower = A_lower @ self.weight
            new_A_upper = A_upper @ self.weight
            node_ndim = 1
            if self.bias is not None:
                bias_lower = A_lower @ self.bias
                bias_upper = A_upper @ self.bias
        else:
            # 1D weight: y = (x * w).sum(-1). Output has no feature dim, so
            # A_lower has shape (*batch, *bounded_out); multiplying in a new
            # trailing dim against ``weight`` gives the predecessor A of shape
            # (*batch, *bounded_out, in_features).
            new_A_lower = A_lower.unsqueeze(-1) * self.weight
            new_A_upper = A_upper.unsqueeze(-1) * self.weight
            node_ndim = 0
            if self.bias is not None:
                bias_lower = A_lower * self.bias
                bias_upper = A_upper * self.bias

        if self.bias is None:
            zero = _zero_bias(A_lower, node_ndim=node_ndim)
            bias_lower = zero
            bias_upper = zero

        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=bias_lower,
            bias_upper=bias_upper,
        )


@final
@dataclass
class MatmulRightConstantRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x @ W`` (right operand constant).

    Parameters
    ----------
    weight : torch.Tensor
        Constant right operand of shape ``(in_features, out_features)``.
    input_node : fx.Node
        The fx graph node for the left (abstract) operand.
    """

    weight: torch.Tensor
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        new_A_lower = A_lower @ self.weight.T
        new_A_upper = A_upper @ self.weight.T

        zero = _zero_bias(A_lower, node_ndim=1)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )


@final
@dataclass
class MatmulBothAbstractRelaxation(BackwardRelaxation):
    """Backward relaxation for ``z = a @ b`` with both operands abstract.

    Uses the McCormick envelope on every bilinear term ``a_ik * b_kj`` and
    reduces over the shared ``K`` axis. Supports the same-node case ``x @ x``
    via :func:`accumulate_a_terms`.

    Parameters
    ----------
    params : PairedParams
        McCormick parameters, each tensor of shape ``(*batch, M, K, N)``.
    left_node : fx.Node
        The fx graph node for the left (``a``) operand.
    right_node : fx.Node
        The fx graph node for the right (``b``) operand.
    """

    params: PairedParams
    left_node: fx.Node
    right_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return list(dict.fromkeys([self.left_node, self.right_node]))

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Sign-decompose downstream A-matrices against per-bilinear-term params.

        ``A_lower``, ``A_upper`` carry shape ``(*batch, *bounded_out, M, N)``
        and the params carry shape ``(*batch, M, K, N)``. We broadcast to the
        joint shape ``(*batch, *bounded_out, M, K, N)`` by inserting a ``K``
        singleton into the A-matrices and bounded-out singletons into the
        params, then contract the result toward each predecessor's natural
        feature layout (``(M, K)`` for ``a``, ``(K, N)`` for ``b``).
        """
        p = self.params
        # Params carry the matmul node's full feature layout (*op_batch, M, K, N).
        # bounded_ndim is what sits between the tape's ``batch_ndim`` and the
        # node feature; A_lower ends with (*op_batch, M, N), i.e. all of the
        # node's feature dims except the K reduction axis.
        params_feature_ndim = p.alpha_lower_a.ndim - batch_ndim
        a_feature_ndim = params_feature_ndim - 1  # (*op_batch, M, N)
        bounded_ndim = A_lower.ndim - batch_ndim - a_feature_ndim

        def bc_params(t: torch.Tensor) -> torch.Tensor:
            """Insert ``bounded_out`` singletons between batch and params feature dims."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        # Insert K axis as singleton into A: (*batch, *bounded_out, M, 1, N).
        A_low_k = A_lower.unsqueeze(-2)
        A_up_k = A_upper.unsqueeze(-2)

        A_l_pos = A_low_k.clamp(min=0)
        A_l_neg = A_low_k.clamp(max=0)
        A_u_pos = A_up_k.clamp(min=0)
        A_u_neg = A_up_k.clamp(max=0)

        alpha_la = bc_params(p.alpha_lower_a)
        alpha_ua = bc_params(p.alpha_upper_a)
        alpha_lb = bc_params(p.alpha_lower_b)
        alpha_ub = bc_params(p.alpha_upper_b)
        gamma_l = bc_params(p.bias_lower)
        gamma_u = bc_params(p.bias_upper)

        # Predecessor A for a: (*batch, *bounded_out, M, K). Sum over N (dim -1).
        new_A_low_a = (A_l_pos * alpha_la + A_l_neg * alpha_ua).sum(dim=-1)
        new_A_up_a = (A_u_pos * alpha_ua + A_u_neg * alpha_la).sum(dim=-1)

        # Predecessor A for b: (*batch, *bounded_out, K, N). Sum over M
        # (dim -3 in the joint (*, M, K, N) shape).
        new_A_low_b = (A_l_pos * alpha_lb + A_l_neg * alpha_ub).sum(dim=-3)
        new_A_up_b = (A_u_pos * alpha_ub + A_u_neg * alpha_lb).sum(dim=-3)

        # Bias delta: sum over M, K, N.
        sum_dims = tuple(range(-params_feature_ndim, 0))
        delta_bias_lower = (A_l_pos * gamma_l + A_l_neg * gamma_u).sum(dim=sum_dims)
        delta_bias_upper = (A_u_pos * gamma_u + A_u_neg * gamma_l).sum(dim=sum_dims)

        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        accumulate_a_terms(a_terms, self.left_node, new_A_low_a, new_A_up_a)
        accumulate_a_terms(a_terms, self.right_node, new_A_low_b, new_A_up_b)

        return BackwardContributions(
            a_terms=a_terms,
            bias_lower=delta_bias_lower,
            bias_upper=delta_bias_upper,
        )


@final
@dataclass
class MatmulLeftConstantRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = W @ x`` (left operand constant).

    Parameters
    ----------
    weight : torch.Tensor
        Constant left operand of shape ``(out_features, in_features)``.
    input_node : fx.Node
        The fx graph node for the right (abstract) operand.
    """

    weight: torch.Tensor
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        new_A_lower = A_lower @ self.weight
        new_A_upper = A_upper @ self.weight

        zero = _zero_bias(A_lower, node_ndim=1)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )


@final
@dataclass
class AddRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x1 + x2`` (both abstract).

    Parameters
    ----------
    left_node : fx.Node
        The fx graph node for the left operand.
    right_node : fx.Node
        The fx graph node for the right operand.
    input_ndim : int
        Rank of the (broadcast) operand tensor, captured at build time so the
        backward pass can compute the per-node feature-dim count without
        touching ``node.meta``.
    """

    left_node: fx.Node
    right_node: fx.Node
    input_ndim: int

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        accumulate_a_terms(a_terms, self.left_node, A_lower, A_upper)
        accumulate_a_terms(a_terms, self.right_node, A_lower, A_upper)

        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower, node_ndim=node_ndim)
        return BackwardContributions(a_terms=a_terms, bias_lower=zero, bias_upper=zero)


@final
@dataclass
class SubRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x1 - x2`` (both abstract).

    Parameters
    ----------
    left_node : fx.Node
        The fx graph node for the left operand.
    right_node : fx.Node
        The fx graph node for the right operand.
    input_ndim : int
        Rank of the (broadcast) operand tensor, captured at build time.
    """

    left_node: fx.Node
    right_node: fx.Node
    input_ndim: int

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        accumulate_a_terms(a_terms, self.left_node, A_lower, A_upper)
        accumulate_a_terms(a_terms, self.right_node, -A_lower, -A_upper)

        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower, node_ndim=node_ndim)
        return BackwardContributions(a_terms=a_terms, bias_lower=zero, bias_upper=zero)


@final
@dataclass
class ConstantAddRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x + c`` (constant addend).

    When ``negate_input`` is ``True``, the A-matrices are negated before
    being passed to the predecessor, implementing ``y = c - x``.

    Parameters
    ----------
    constant : torch.Tensor
        The constant tensor being added.
    input_node : fx.Node
        The fx graph node for the abstract operand.
    negate_input : bool
        If ``True``, negate the A-matrices for the predecessor (for ``c - x``).
    """

    constant: torch.Tensor
    input_node: fx.Node
    negate_input: bool = False

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        node_ndim = self.constant.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - self.constant.ndim

        c_bc = self.constant.reshape(
            self.constant.shape[:batch_ndim] + (1,) * bounded_ndim + self.constant.shape[batch_ndim:]
        )

        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        if sum_dims:
            delta_lower = (A_lower * c_bc).sum(dim=sum_dims)
            delta_upper = (A_upper * c_bc).sum(dim=sum_dims)
        else:
            delta_lower = A_lower * c_bc
            delta_upper = A_upper * c_bc

        if self.negate_input:
            pred_A_lower, pred_A_upper = -A_lower, -A_upper
        else:
            pred_A_lower, pred_A_upper = A_lower, A_upper

        return BackwardContributions(
            a_terms={self.input_node: (pred_A_lower, pred_A_upper)},
            bias_lower=delta_lower,
            bias_upper=delta_upper,
        )


@final
@dataclass
class NegRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = -x``.

    Parameters
    ----------
    input_node : fx.Node
        The fx graph node for the input operand.
    input_ndim : int
        Rank of the input tensor, captured at build time.
    """

    input_node: fx.Node
    input_ndim: int

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower, node_ndim=node_ndim)
        return BackwardContributions(
            a_terms={self.input_node: (-A_lower, -A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )


@final
@dataclass
class ScaleRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = c * x`` (element-wise constant scale).

    Parameters
    ----------
    scale : torch.Tensor
        The constant scale tensor.
    input_node : fx.Node
        The fx graph node for the abstract operand.
    """

    scale: torch.Tensor
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        bounded_ndim = A_lower.ndim - self.scale.ndim
        c = self.scale.reshape(self.scale.shape[:batch_ndim] + (1,) * bounded_ndim + self.scale.shape[batch_ndim:])

        new_A_lower = A_lower * c
        new_A_upper = A_upper * c

        node_ndim = self.scale.ndim - batch_ndim
        zero = _zero_bias(A_lower, node_ndim=node_ndim)
        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=zero,
            bias_upper=zero,
        )
