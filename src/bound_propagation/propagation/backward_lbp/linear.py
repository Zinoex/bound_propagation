"""Backward LBP strategies and relaxations for linear / affine operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final
from plum import dispatch

from ...bounds import IntervalBounds
from ...linear_operators import DenseOperator, IdentityOperator, LinearOperator
from ..linear_relaxations.alpha_resolvers import resolve_matmul_etas
from ..linear_relaxations.pairwise import PairedParams, compute_mul_relaxation
from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    _wrap_a_term_tensors,
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
                left_shape=tuple(tape.shape_of(node.args[0])),  # ty:ignore[invalid-argument-type]
                right_shape=tuple(tape.shape_of(node.args[1])),  # ty:ignore[invalid-argument-type]
                input_ndim=len(tape.shape_of(node)),
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
                left_shape=tuple(tape.shape_of(node.args[0])),  # ty:ignore[invalid-argument-type]
                right_shape=tuple(tape.shape_of(node.args[1])),  # ty:ignore[invalid-argument-type]
                input_ndim=len(tape.shape_of(node)),
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


def _reduce_input_to_shape(op: LinearOperator, target_shape: tuple[int, ...] | torch.Size) -> LinearOperator:
    """Sum / squeeze ``op``'s input axes so ``input_shape`` becomes ``target_shape``.

    Backward-LBP correctness for broadcast operations (``y = a + b`` with
    ``a.shape != b.shape``): the upstream A has ``input_shape == y.shape``,
    but each predecessor's accumulated A must have ``input_shape == predecessor.shape``.
    Where the predecessor had a size-1 axis (or no axis at all) that was
    broadcast to a larger axis in ``y``, A's contribution to that predecessor
    is the *sum* of A across that broadcast axis.

    Currently materializes to dense and reduces; future structured operators
    (e.g. ``Patches``) can override this without changing the call sites.
    """
    source_shape = tuple(op.input_shape)
    target = tuple(target_shape)
    if source_shape == target:
        return op

    ndim_diff = len(source_shape) - len(target)
    if ndim_diff < 0:
        raise ValueError(f"_reduce_input_to_shape: cannot grow input rank from {source_shape} to {target}")
    padded_target = (1,) * ndim_diff + target

    sum_axes: list[int] = []
    for i, (s, p) in enumerate(zip(source_shape, padded_target, strict=True)):
        if s != p:
            if p != 1:
                raise ValueError(
                    f"_reduce_input_to_shape: source {source_shape} not broadcast-compatible "
                    f"with target {target} (axis {i}: {s} vs {p})"
                )
            sum_axes.append(op.output_ndim + i)

    tensor = op.to_dense().tensor
    if sum_axes:
        tensor = tensor.sum(dim=sum_axes, keepdim=True)
    if ndim_diff > 0:
        leading_output = tensor.shape[: op.output_ndim]
        tensor = tensor.reshape(*leading_output, *target)
    return DenseOperator(tensor, output_shape=op.output_shape)


def _broadcast_constant(constant: torch.Tensor, *, batch_ndim: int, bounded_ndim: int, node_ndim: int) -> torch.Tensor:
    """Reshape ``constant`` to broadcast against ``A`` of shape ``(*batch, *bounded, *node)``.

    Inserts ``bounded_ndim`` singleton dims between the constant's batch and
    feature axes, then prepends extra singleton feature dims if the constant
    has fewer feature axes than ``node_ndim`` (e.g. a scalar constant
    multiplying a multi-feature node). This avoids assuming the constant has
    the same rank as the node — scalars and partial-broadcast constants
    were previously misclassified, leading to wrong-shape bias contributions.
    """
    if constant.ndim < batch_ndim:
        raise ValueError(f"constant rank {constant.ndim} is smaller than batch rank {batch_ndim}")
    feature_shape = tuple(constant.shape[batch_ndim:])
    pad_for_features = node_ndim - len(feature_shape)
    if pad_for_features < 0:
        raise ValueError(f"constant feature rank {len(feature_shape)} exceeds input feature rank {node_ndim}")
    new_shape = tuple(constant.shape[:batch_ndim]) + (1,) * bounded_ndim + (1,) * pad_for_features + feature_shape
    return constant.reshape(new_shape)


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

    @dispatch
    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        output_shape = A_lower.output_shape
        A_lower_t = A_lower.to_dense().tensor
        A_upper_t = A_upper.to_dense().tensor

        if self.weight.ndim == 2:
            # A: (*batch, *bounded_out, out_features)
            # weight: (out_features, in_features) -> A @ weight gives (..., in_features)
            new_A_lower = A_lower_t @ self.weight
            new_A_upper = A_upper_t @ self.weight
            node_ndim = 1
            if self.bias is not None:
                bias_lower = A_lower_t @ self.bias
                bias_upper = A_upper_t @ self.bias
        else:
            # 1D weight: y = (x * w).sum(-1). Output has no feature dim, so
            # A_lower has shape (*batch, *bounded_out); multiplying in a new
            # trailing dim against ``weight`` gives the predecessor A of shape
            # (*batch, *bounded_out, in_features).
            new_A_lower = A_lower_t.unsqueeze(-1) * self.weight
            new_A_upper = A_upper_t.unsqueeze(-1) * self.weight
            node_ndim = 0
            if self.bias is not None:
                bias_lower = A_lower_t * self.bias
                bias_upper = A_upper_t * self.bias

        if self.bias is None:
            zero = _zero_bias(A_lower_t, node_ndim=node_ndim)
            bias_lower = zero
            bias_upper = zero

        return BackwardContributions(
            a_terms=_wrap_a_term_tensors({self.input_node: (new_A_lower, new_A_upper)}, len(output_shape)),
            bias_lower=bias_lower,
            bias_upper=bias_upper,
        )

    @dispatch
    def backward_through(  # noqa: F811
        self, A_lower: IdentityOperator, A_upper: IdentityOperator, batch_ndim: int
    ) -> BackwardContributions:
        """Identity @ Linear = the weight itself, reshaped into A's output axes.

        Avoids materializing any ``eye(numel)``: the resulting A is simply
        ``weight`` (or ``weight^T`` for a 1D reduction) broadcast to the
        caller's batch axes; the bias contribution is ``bias`` broadcast the
        same way.
        """
        output_shape = A_lower.output_shape
        batch_shape = A_lower.batch_shape
        leading_ones = (1,) * len(batch_shape)

        if self.weight.ndim == 2:
            out_features, in_features = self.weight.shape
            # Weight reshaped to (*batch_ones, out_features, in_features); output_shape
            # leads are (*batch_shape, out_features) so the dense op holds Identity@weight.
            weight_tensor = self.weight.reshape(*leading_ones, out_features, in_features)
            new_A_op: LinearOperator = DenseOperator(weight_tensor, output_shape=output_shape)
            if self.bias is not None:
                bias_tensor = self.bias.reshape(*leading_ones, out_features).expand(output_shape)
                bias_lower: torch.Tensor = bias_tensor
                bias_upper: torch.Tensor = bias_tensor
            else:
                bias_lower = torch.zeros(output_shape, dtype=self.weight.dtype, device=self.weight.device)
                bias_upper = bias_lower
        else:
            # 1D weight: linear reduces last feature dim. Feature_shape must be () at A.
            (in_features,) = self.weight.shape
            weight_tensor = self.weight.reshape(*leading_ones, in_features)
            new_A_op = DenseOperator(weight_tensor, output_shape=output_shape)
            if self.bias is not None:
                bias_scalar = self.bias.reshape(*leading_ones).expand(output_shape)
                bias_lower = bias_scalar
                bias_upper = bias_scalar
            else:
                bias_lower = torch.zeros(output_shape, dtype=self.weight.dtype, device=self.weight.device)
                bias_upper = bias_lower

        return BackwardContributions(
            a_terms={self.input_node: (new_A_op, new_A_op)},
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

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        output_shape = A_lower.output_shape
        A_lower_t = A_lower.to_dense().tensor
        A_upper_t = A_upper.to_dense().tensor

        new_A_lower = A_lower_t @ self.weight.T
        new_A_upper = A_upper_t @ self.weight.T

        zero = _zero_bias(A_lower_t, node_ndim=1)
        return BackwardContributions(
            a_terms=_wrap_a_term_tensors({self.input_node: (new_A_lower, new_A_upper)}, len(output_shape)),
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
        A_lower: LinearOperator,
        A_upper: LinearOperator,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Sign-decompose downstream A-matrices against per-bilinear-term params."""
        output_shape = A_lower.output_shape

        p = self.params
        params_feature_ndim = p.alpha_lower_a.ndim - batch_ndim
        a_feature_ndim = params_feature_ndim - 1  # (*op_batch, M, N)
        bounded_ndim = A_lower.output_ndim + A_lower.input_ndim - batch_ndim - a_feature_ndim

        def bc_params(t: torch.Tensor) -> torch.Tensor:
            """Insert ``bounded_out`` singletons between batch and params feature dims."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        # Insert K axis as singleton into A: (*batch, *bounded_out, M, 1, N).
        # Clamp commutes with unsqueeze, so route the sign decomposition through the
        # operator API (lets structured A's specialize) and then unsqueeze the result.
        A_l_pos = A_lower.clamp_min(0).to_dense().tensor.unsqueeze(-2)
        A_l_neg = A_lower.clamp_max(0).to_dense().tensor.unsqueeze(-2)
        A_u_pos = A_upper.clamp_min(0).to_dense().tensor.unsqueeze(-2)
        A_u_neg = A_upper.clamp_max(0).to_dense().tensor.unsqueeze(-2)

        alpha_la = bc_params(p.alpha_lower_a)
        alpha_ua = bc_params(p.alpha_upper_a)
        alpha_lb = bc_params(p.alpha_lower_b)
        alpha_ub = bc_params(p.alpha_upper_b)
        gamma_l = bc_params(p.bias_lower)
        gamma_u = bc_params(p.bias_upper)

        new_A_low_a = (A_l_pos * alpha_la + A_l_neg * alpha_ua).sum(dim=-1)
        new_A_up_a = (A_u_pos * alpha_ua + A_u_neg * alpha_la).sum(dim=-1)

        new_A_low_b = (A_l_pos * alpha_lb + A_l_neg * alpha_ub).sum(dim=-3)
        new_A_up_b = (A_u_pos * alpha_ub + A_u_neg * alpha_lb).sum(dim=-3)

        sum_dims = tuple(range(-params_feature_ndim, 0))
        delta_bias_lower = (A_l_pos * gamma_l + A_l_neg * gamma_u).sum(dim=sum_dims)
        delta_bias_upper = (A_u_pos * gamma_u + A_u_neg * gamma_l).sum(dim=sum_dims)

        out_ndim = len(output_shape)
        a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
        accumulate_a_terms(
            a_terms,
            self.left_node,
            DenseOperator(new_A_low_a, output_shape=torch.Size(new_A_low_a.shape[:out_ndim])),
            DenseOperator(new_A_up_a, output_shape=torch.Size(new_A_up_a.shape[:out_ndim])),
        )
        accumulate_a_terms(
            a_terms,
            self.right_node,
            DenseOperator(new_A_low_b, output_shape=torch.Size(new_A_low_b.shape[:out_ndim])),
            DenseOperator(new_A_up_b, output_shape=torch.Size(new_A_up_b.shape[:out_ndim])),
        )

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

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        output_shape = A_lower.output_shape
        A_lower_t = A_lower.to_dense().tensor
        A_upper_t = A_upper.to_dense().tensor

        new_A_lower = A_lower_t @ self.weight
        new_A_upper = A_upper_t @ self.weight

        zero = _zero_bias(A_lower_t, node_ndim=1)
        return BackwardContributions(
            a_terms=_wrap_a_term_tensors({self.input_node: (new_A_lower, new_A_upper)}, len(output_shape)),
            bias_lower=zero,
            bias_upper=zero,
        )


@final
@dataclass
class AddRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x1 + x2`` (both abstract).

    Parameters
    ----------
    left_node, right_node : fx.Node
        The fx graph nodes for the operands.
    left_shape, right_shape : tuple[int, ...]
        Operand shapes captured at build time. Needed to reduce A across
        broadcast axes when ``left.shape != right.shape`` — A on the
        upstream side has ``input_shape == result.shape``, but each
        predecessor's accumulated A must have ``input_shape == predecessor.shape``.
    input_ndim : int
        Rank of the broadcast result tensor, captured at build time so the
        backward pass can compute the per-node feature-dim count without
        touching ``node.meta``.
    """

    left_node: fx.Node
    right_node: fx.Node
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]
    input_ndim: int

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    @dispatch
    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        A_lower_t = A_lower.to_dense().tensor

        a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
        # Reduce A to each predecessor's shape (handles broadcast via sum).
        left_lower = _reduce_input_to_shape(A_lower, self.left_shape[batch_ndim:])
        left_upper = _reduce_input_to_shape(A_upper, self.left_shape[batch_ndim:])
        right_lower = _reduce_input_to_shape(A_lower, self.right_shape[batch_ndim:])
        right_upper = _reduce_input_to_shape(A_upper, self.right_shape[batch_ndim:])
        accumulate_a_terms(a_terms, self.left_node, left_lower, left_upper)
        accumulate_a_terms(a_terms, self.right_node, right_lower, right_upper)

        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower_t, node_ndim=node_ndim)
        return BackwardContributions(a_terms=a_terms, bias_lower=zero, bias_upper=zero)

    @dispatch
    def backward_through(  # noqa: F811
        self, A_lower: IdentityOperator, A_upper: IdentityOperator, batch_ndim: int
    ) -> BackwardContributions:
        """Identity passes through Add for each distinct predecessor, with
        broadcast-reduction when operand shapes differ.

        Skips the :meth:`to_dense` materialization needed to compute the zero
        bias shape (we use ``A_lower.output_shape`` directly, which is
        invariant through this relaxation).
        """
        a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
        left_lower = _reduce_input_to_shape(A_lower, self.left_shape[batch_ndim:])
        left_upper = _reduce_input_to_shape(A_upper, self.left_shape[batch_ndim:])
        right_lower = _reduce_input_to_shape(A_lower, self.right_shape[batch_ndim:])
        right_upper = _reduce_input_to_shape(A_upper, self.right_shape[batch_ndim:])
        accumulate_a_terms(a_terms, self.left_node, left_lower, left_upper)
        accumulate_a_terms(a_terms, self.right_node, right_lower, right_upper)

        zero = torch.zeros(A_lower.output_shape, dtype=A_lower.dtype, device=A_lower.device)
        return BackwardContributions(a_terms=a_terms, bias_lower=zero, bias_upper=zero)


@final
@dataclass
class SubRelaxation(BackwardRelaxation):
    """Backward relaxation for ``y = x1 - x2`` (both abstract).

    Parameters
    ----------
    left_node, right_node : fx.Node
        The fx graph nodes for the operands.
    left_shape, right_shape : tuple[int, ...]
        Operand shapes captured at build time (see :class:`AddRelaxation`).
    input_ndim : int
        Rank of the broadcast result tensor, captured at build time.
    """

    left_node: fx.Node
    right_node: fx.Node
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]
    input_ndim: int

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        A_lower_t = A_lower.to_dense().tensor

        a_terms: dict[fx.Node, tuple[LinearOperator, LinearOperator]] = {}
        # Reduce A to each predecessor's shape (handles broadcast via sum).
        # Right operand contributes ``-A``: negate after reduction (commutes with sum).
        left_lower = _reduce_input_to_shape(A_lower, self.left_shape[batch_ndim:])
        left_upper = _reduce_input_to_shape(A_upper, self.left_shape[batch_ndim:])
        right_lower = _reduce_input_to_shape(A_lower, self.right_shape[batch_ndim:]).neg()
        right_upper = _reduce_input_to_shape(A_upper, self.right_shape[batch_ndim:]).neg()
        accumulate_a_terms(a_terms, self.left_node, left_lower, left_upper)
        accumulate_a_terms(a_terms, self.right_node, right_lower, right_upper)

        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower_t, node_ndim=node_ndim)
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

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        A_lower_t = A_lower.to_dense().tensor
        A_upper_t = A_upper.to_dense().tensor

        # Derive shape metadata from the operator (not the constant) so a
        # scalar / partially-broadcast constant aligns with the propagated A's
        # input axes correctly.
        node_ndim = A_lower.input_ndim
        bounded_ndim = A_lower.output_ndim - batch_ndim

        # Reshape the constant to broadcast against ``(*batch, *bounded_out, *node)``.
        c_bc = _broadcast_constant(self.constant, batch_ndim=batch_ndim, bounded_ndim=bounded_ndim, node_ndim=node_ndim)

        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        if sum_dims:
            delta_lower = (A_lower_t * c_bc).sum(dim=sum_dims)
            delta_upper = (A_upper_t * c_bc).sum(dim=sum_dims)
        else:
            delta_lower = A_lower_t * c_bc
            delta_upper = A_upper_t * c_bc

        if self.negate_input:
            pred_A_lower, pred_A_upper = A_lower.neg(), A_upper.neg()
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

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        A_lower_t = A_lower.to_dense().tensor

        node_ndim = _node_feature_ndim(self.input_ndim, batch_ndim)
        zero = _zero_bias(A_lower_t, node_ndim=node_ndim)
        return BackwardContributions(
            a_terms={self.input_node: (A_lower.neg(), A_upper.neg())},
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

    def backward_through(
        self, A_lower: LinearOperator, A_upper: LinearOperator, batch_ndim: int
    ) -> BackwardContributions:
        output_shape = A_lower.output_shape
        A_lower_t = A_lower.to_dense().tensor
        A_upper_t = A_upper.to_dense().tensor

        # Derive shape metadata from the operator (not the scale tensor) so a
        # scalar / partially-broadcast scale aligns with the propagated A's
        # input axes correctly.
        node_ndim = A_lower.input_ndim
        bounded_ndim = A_lower.output_ndim - batch_ndim

        c = _broadcast_constant(self.scale, batch_ndim=batch_ndim, bounded_ndim=bounded_ndim, node_ndim=node_ndim)

        new_A_lower = A_lower_t * c
        new_A_upper = A_upper_t * c

        zero = _zero_bias(A_lower_t, node_ndim=node_ndim)
        return BackwardContributions(
            a_terms=_wrap_a_term_tensors({self.input_node: (new_A_lower, new_A_upper)}, len(output_shape)),
            bias_lower=zero,
            bias_upper=zero,
        )
