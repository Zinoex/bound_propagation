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


@final
@dataclass
class SymbolicPairedLinearRelaxation(SymbolicLinearRelaxation):
    concrete_relaxation: PairedLinearRelaxation

    input_left: SymbolicLinearRelaxation
    input_right: SymbolicLinearRelaxation

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        r = self.concrete_relaxation
        node_ndim = r.coeffs_lower[0].ndim - batch_ndim
        bounded_ndim = A_lower.ndim - r.coeffs_lower[0].ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast ``(*batch, *node)`` → ``(*batch, *bounded_out, *node)``."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        # Left input: sign decomposition on coeffs[0]
        new_A_lower_left = A_l_pos * bc(r.coeffs_lower[0]) + A_l_neg * bc(r.coeffs_upper[0])
        new_A_upper_left = A_u_pos * bc(r.coeffs_upper[0]) + A_u_neg * bc(r.coeffs_lower[0])
        bounds_left = self.input_left.backward(new_A_lower_left, new_A_upper_left, batch_ndim)

        # Right input: sign decomposition on coeffs[1]
        new_A_lower_right = A_l_pos * bc(r.coeffs_lower[1]) + A_l_neg * bc(r.coeffs_upper[1])
        new_A_upper_right = A_u_pos * bc(r.coeffs_upper[1]) + A_u_neg * bc(r.coeffs_lower[1])
        bounds_right = self.input_right.backward(new_A_lower_right, new_A_upper_right, batch_ndim)

        # Bias contribution: sum over the trailing node dimensions.
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_bias_lower = A_l_pos * bc(r.bias_lower) + A_l_neg * bc(r.bias_upper)
        delta_bias_upper = A_u_pos * bc(r.bias_upper) + A_u_neg * bc(r.bias_lower)
        if sum_dims:
            delta_bias_lower = delta_bias_lower.sum(dim=sum_dims)
            delta_bias_upper = delta_bias_upper.sum(dim=sum_dims)

        bias_lower = bounds_left.bias_lower + bounds_right.bias_lower + delta_bias_lower
        bias_upper = bounds_left.bias_upper + bounds_right.bias_upper + delta_bias_upper

        # Merge linear contributions by input_id (handles shared regions between left and right)
        merged: dict[int, tuple[SimpleRegion, torch.Tensor, torch.Tensor]] = {}
        ordered_ids: list[int] = []

        for bounds in [bounds_left, bounds_right]:
            for iid, region, wl, wu in zip(
                bounds.input_ids, bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True
            ):
                if iid in merged:
                    merged[iid] = (merged[iid][0], merged[iid][1] + wl, merged[iid][2] + wu)
                else:
                    ordered_ids.append(iid)
                    merged[iid] = (region, wl, wu)

        regions = [merged[iid][0] for iid in ordered_ids]
        linear_lower = [merged[iid][1] for iid in ordered_ids]
        linear_upper = [merged[iid][2] for iid in ordered_ids]

        return LinearBounds(
            regions=regions,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=ordered_ids or None,
        )
