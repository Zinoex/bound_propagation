"""Backward LBP strategies for linear / affine operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ..linear_relaxations.base import (
    SymbolicIntervalLeaf,
    SymbolicLinearRelaxation,
)
from ..linear_relaxations.linear import (
    SymbolicAddBounds,
    SymbolicConstantAdd,
    SymbolicLinear,
    SymbolicMatmulLeftConstant,
    SymbolicMatmulRightConstant,
    SymbolicNeg,
    SymbolicScale,
    SymbolicSubBounds,
)
from .base import BackwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class BackwardLBPLinear(BackwardLBPStrategy):
    """Backward LBP strategy for nn.Linear / F.linear."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, kwargs = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPLinear requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

        if weight is None:
            raise ValueError("BackwardLBPLinear requires a weight tensor")

        return SymbolicLinear(weight=weight, bias=bias, input=sym_input)


class BackwardLBPMatmul(BackwardLBPStrategy):
    """Backward LBP strategy for matmul (abstract@constant or constant@abstract)."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, torch.Tensor):
            return SymbolicMatmulRightConstant(weight=right, input=left)

        if isinstance(left, torch.Tensor) and isinstance(right, SymbolicLinearRelaxation):
            return SymbolicMatmulLeftConstant(weight=left, input=right)

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            raise NotImplementedError("Backward LBP matmul with two abstract operands is not supported")

        raise TypeError(
            f"BackwardLBPMatmul requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )


class BackwardLBPAdd(BackwardLBPStrategy):
    """Backward LBP strategy for addition."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return SymbolicAddBounds(input_left=left, input_right=right)

        if isinstance(left, SymbolicLinearRelaxation):
            constant = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"])
            return SymbolicConstantAdd(constant=constant, input=left)

        if isinstance(right, SymbolicLinearRelaxation):
            constant = torch.as_tensor(left, dtype=node.meta["tensor_meta"]["dtype"])
            return SymbolicConstantAdd(constant=constant, input=right)

        raise TypeError(
            f"BackwardLBPAdd requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )


class BackwardLBPSub(BackwardLBPStrategy):
    """Backward LBP strategy for subtraction."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return SymbolicSubBounds(input_left=left, input_right=right)

        if isinstance(left, SymbolicLinearRelaxation):
            constant = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"])
            return SymbolicConstantAdd(constant=-constant, input=left)

        if isinstance(right, SymbolicLinearRelaxation):
            # c - x = -(x - c) = -(x) + c
            constant = torch.as_tensor(left, dtype=node.meta["tensor_meta"]["dtype"])
            return SymbolicConstantAdd(constant=constant, input=SymbolicNeg(input=right))

        raise TypeError(
            f"BackwardLBPSub requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )


class BackwardLBPNeg(BackwardLBPStrategy):
    """Backward LBP strategy for negation."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        sym_input = args[0]
        if not isinstance(sym_input, SymbolicLinearRelaxation):
            raise TypeError(f"BackwardLBPNeg requires input to be SymbolicLinearRelaxation, got {type(sym_input)}")
        return SymbolicNeg(input=sym_input)


class BackwardLBPMul(BackwardLBPStrategy):
    """Backward LBP strategy for multiplication (abstract*constant or abstract*abstract)."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return self._mul_abstract(node, left, right)

        if isinstance(left, SymbolicLinearRelaxation):
            scale = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"]).expand(
                node.meta["tensor_meta"]["shape"]
            )
            return SymbolicScale(scale=scale, input=left)

        if isinstance(right, SymbolicLinearRelaxation):
            scale = torch.as_tensor(left, dtype=node.meta["tensor_meta"]["dtype"]).expand(
                node.meta["tensor_meta"]["shape"]
            )
            return SymbolicScale(scale=scale, input=right)

        raise TypeError(
            f"BackwardLBPMul requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )

    def _mul_abstract(
        self,
        node: fx.Node,
        left: SymbolicLinearRelaxation,
        right: SymbolicLinearRelaxation,
    ) -> SymbolicLinearRelaxation:
        """Abstract * abstract: McCormick relaxation via PairedLinearRelaxation."""
        from ..linear_relaxations.mul import compute_mul_relaxation
        from .base import concretize_symbolic

        left_node = node.args[0]
        right_node = node.args[1]
        left_shape = left_node.meta["tensor_meta"]["shape"]
        right_shape = right_node.meta["tensor_meta"]["shape"]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        la, ua = concretize_symbolic(left, left_shape, dtype, device)
        lb, ub = concretize_symbolic(right, right_shape, dtype, device)

        relaxation = compute_mul_relaxation(la, ua, lb, ub)
        return relaxation.symbolic_forward([left, right])


class BackwardLBPDiv(BackwardLBPStrategy):
    """Backward LBP strategy for division (abstract/constant, constant/abstract, abstract/abstract)."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return self._div_abstract(node, left, right)

        if isinstance(left, SymbolicLinearRelaxation) and not isinstance(right, SymbolicLinearRelaxation):
            # abstract / constant = abstract * (1/constant)
            divisor = torch.as_tensor(right, dtype=node.meta["tensor_meta"]["dtype"]).expand(
                node.meta["tensor_meta"]["shape"]
            )
            return SymbolicScale(scale=1.0 / divisor, input=left)

        if isinstance(right, SymbolicLinearRelaxation) and not isinstance(left, SymbolicLinearRelaxation):
            return self._constant_div_abstract(node, left, right)

        raise TypeError(
            f"BackwardLBPDiv requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )

    def _div_abstract(
        self,
        node: fx.Node,
        left: SymbolicLinearRelaxation,
        right: SymbolicLinearRelaxation,
    ) -> SymbolicLinearRelaxation:
        """Abstract / abstract: decompose as a * (1/b) via relaxation."""
        from ..linear_relaxations.div import compute_div_relaxation
        from .base import concretize_symbolic

        left_node = node.args[0]
        right_node = node.args[1]
        left_shape = left_node.meta["tensor_meta"]["shape"]
        right_shape = right_node.meta["tensor_meta"]["shape"]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        la, ua = concretize_symbolic(left, left_shape, dtype, device)
        lb, ub = concretize_symbolic(right, right_shape, dtype, device)

        relaxation = compute_div_relaxation(la, ua, lb, ub)
        return relaxation.symbolic_forward([left, right])

    def _constant_div_abstract(
        self,
        node: fx.Node,
        constant: object,
        right: SymbolicLinearRelaxation,
    ) -> SymbolicLinearRelaxation:
        """Constant / abstract: use constant_div relaxation."""
        from ..linear_relaxations.constant_div import compute_constant_div_relaxation
        from .base import concretize_symbolic

        right_node = node.args[1]
        right_shape = right_node.meta["tensor_meta"]["shape"]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        lower_x, upper_x = concretize_symbolic(right, right_shape, dtype, device)
        relaxation = compute_constant_div_relaxation(lower_x, upper_x, constant)
        return relaxation.symbolic_forward([right])


class BackwardLBPMaximum(BackwardLBPStrategy):
    """Backward LBP strategy for element-wise maximum."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:

        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return self._max_abstract(node, left, right)

        if isinstance(left, SymbolicLinearRelaxation):
            return self._max_with_constant(node, left, right, sym_is_left=True)

        if isinstance(right, SymbolicLinearRelaxation):
            return self._max_with_constant(node, right, left, sym_is_left=False)

        raise TypeError(
            f"BackwardLBPMaximum requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )

    def _max_abstract(
        self,
        node: fx.Node,
        left: SymbolicLinearRelaxation,
        right: SymbolicLinearRelaxation,
    ) -> SymbolicLinearRelaxation:
        from ..linear_relaxations.maximum import compute_maximum_relaxation
        from .base import concretize_symbolic

        left_node = node.args[0]
        right_node = node.args[1]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        la, ua = concretize_symbolic(left, left_node.meta["tensor_meta"]["shape"], dtype, device)
        lb, ub = concretize_symbolic(right, right_node.meta["tensor_meta"]["shape"], dtype, device)

        relaxation = compute_maximum_relaxation(la, ua, lb, ub)
        return relaxation.symbolic_forward([left, right])

    def _max_with_constant(
        self,
        node: fx.Node,
        sym: SymbolicLinearRelaxation,
        constant: object,
        sym_is_left: bool,
    ) -> SymbolicLinearRelaxation:
        from ..linear_relaxations.maximum import compute_maximum_relaxation
        from .base import concretize_symbolic

        sym_node = node.args[0] if sym_is_left else node.args[1]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        ls, us = concretize_symbolic(sym, sym_node.meta["tensor_meta"]["shape"], dtype, device)
        c = torch.as_tensor(constant, dtype=dtype, device=device).expand_as(ls)

        if sym_is_left:
            relaxation = compute_maximum_relaxation(ls, us, c, c)
        else:
            relaxation = compute_maximum_relaxation(c, c, ls, us)

        # The constant input needs a SymbolicIntervalLeaf
        const_sym = SymbolicIntervalLeaf(lower=c, upper=c)
        if sym_is_left:
            return relaxation.symbolic_forward([sym, const_sym])
        return relaxation.symbolic_forward([const_sym, sym])


class BackwardLBPMinimum(BackwardLBPStrategy):
    """Backward LBP strategy for element-wise minimum."""

    def build_symbolic(self, node: fx.Node, ctx: PropagationContext) -> SymbolicLinearRelaxation:
        args, _ = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, SymbolicLinearRelaxation) and isinstance(right, SymbolicLinearRelaxation):
            return self._min_abstract(node, left, right)

        if isinstance(left, SymbolicLinearRelaxation):
            return self._min_with_constant(node, left, right, sym_is_left=True)

        if isinstance(right, SymbolicLinearRelaxation):
            return self._min_with_constant(node, right, left, sym_is_left=False)

        raise TypeError(
            f"BackwardLBPMinimum requires at least one SymbolicLinearRelaxation, got {type(left)} and {type(right)}"
        )

    def _min_abstract(
        self,
        node: fx.Node,
        left: SymbolicLinearRelaxation,
        right: SymbolicLinearRelaxation,
    ) -> SymbolicLinearRelaxation:
        from ..linear_relaxations.minimum import compute_minimum_relaxation
        from .base import concretize_symbolic

        left_node = node.args[0]
        right_node = node.args[1]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        la, ua = concretize_symbolic(left, left_node.meta["tensor_meta"]["shape"], dtype, device)
        lb, ub = concretize_symbolic(right, right_node.meta["tensor_meta"]["shape"], dtype, device)

        relaxation = compute_minimum_relaxation(la, ua, lb, ub)
        return relaxation.symbolic_forward([left, right])

    def _min_with_constant(
        self,
        node: fx.Node,
        sym: SymbolicLinearRelaxation,
        constant: object,
        sym_is_left: bool,
    ) -> SymbolicLinearRelaxation:
        from ..linear_relaxations.minimum import compute_minimum_relaxation
        from .base import concretize_symbolic

        sym_node = node.args[0] if sym_is_left else node.args[1]
        dtype = node.meta["tensor_meta"]["dtype"]
        device = node.meta.get("device", "cpu")

        ls, us = concretize_symbolic(sym, sym_node.meta["tensor_meta"]["shape"], dtype, device)
        c = torch.as_tensor(constant, dtype=dtype, device=device).expand_as(ls)

        if sym_is_left:
            relaxation = compute_minimum_relaxation(ls, us, c, c)
        else:
            relaxation = compute_minimum_relaxation(c, c, ls, us)

        const_sym = SymbolicIntervalLeaf(lower=c, upper=c)
        if sym_is_left:
            return relaxation.symbolic_forward([sym, const_sym])
        return relaxation.symbolic_forward([const_sym, sym])
