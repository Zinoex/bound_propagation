"""Backward LBP strategies and relaxations for linear / affine operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

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
            # TODO: Implement this case. It is absolutely crucial for handling e.g. Jacobian terms.
            raise NotImplementedError("Backward LBP matmul with two abstract operands is not supported")

        raise TypeError(
            f"BackwardLBPMatmul requires at least one BackwardRelaxation operand, got {type(left)} and {type(right)}"
        )


class BackwardLBPAdd(BackwardLBPStrategy):
    """Backward LBP strategy for addition."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        args, _ = tape.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, BackwardRelaxation) and isinstance(right, BackwardRelaxation):
            return AddRelaxation(left_node=node.args[0], right_node=node.args[1])

        if isinstance(left, BackwardRelaxation):
            constant = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"])
            return ConstantAddRelaxation(constant=constant, input_node=node.args[0])

        if isinstance(right, BackwardRelaxation):
            constant = torch.as_tensor(left, dtype=node.meta["tensor_meta"]["dtype"])
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
            return SubRelaxation(left_node=node.args[0], right_node=node.args[1])

        if isinstance(left, BackwardRelaxation):
            # x - c = x + (-c)
            constant = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"])
            return ConstantAddRelaxation(constant=-constant, input_node=node.args[0])

        if isinstance(right, BackwardRelaxation):
            # c - x: bias gets +c contribution, A gets negated for x
            constant = torch.as_tensor(left, dtype=node.meta["tensor_meta"]["dtype"])
            return ConstantAddRelaxation(constant=constant, input_node=node.args[1], negate_input=True)

        raise TypeError(
            f"BackwardLBPSub requires at least one BackwardRelaxation operand, got {type(left)} and {type(right)}"
        )


class BackwardLBPNeg(BackwardLBPStrategy):
    """Backward LBP strategy for negation."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> BackwardRelaxation:
        return NegRelaxation(input_node=node.args[0])


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


def _node_ndim_from_meta(node: fx.Node, batch_ndim: int) -> int:
    """Infer the number of non-batch feature dimensions from node metadata.

    Falls back to 0 when ``tensor_meta`` is unavailable.
    """
    meta = node.meta.get("tensor_meta")
    if meta is not None:
        return len(meta["shape"]) - batch_ndim
    return 0


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
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.Tensor | None
        Bias vector of shape ``(out_features,)``, or ``None``.
    input_node : fx.Node
        The fx graph node for the input to this linear layer.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        # A: (*batch, *bounded_out, out_features)
        # weight: (out_features, in_features) -> A @ weight gives (..., in_features)
        new_A_lower = A_lower @ self.weight
        new_A_upper = A_upper @ self.weight

        if self.bias is not None:
            bias_lower = A_lower @ self.bias
            bias_upper = A_upper @ self.bias
        else:
            zero = _zero_bias(A_lower, node_ndim=1)
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
    """

    left_node: fx.Node
    right_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        accumulate_a_terms(a_terms, self.left_node, A_lower, A_upper)
        accumulate_a_terms(a_terms, self.right_node, A_lower, A_upper)

        node_ndim = _node_ndim_from_meta(self.left_node, batch_ndim)
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
    """

    left_node: fx.Node
    right_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return list({self.left_node, self.right_node})

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        a_terms: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}
        accumulate_a_terms(a_terms, self.left_node, A_lower, A_upper)
        accumulate_a_terms(a_terms, self.right_node, -A_lower, -A_upper)

        node_ndim = _node_ndim_from_meta(self.left_node, batch_ndim)
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
    """

    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> BackwardContributions:
        node_ndim = _node_ndim_from_meta(self.input_node, batch_ndim)
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
