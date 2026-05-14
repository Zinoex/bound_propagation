from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds, LinearBounds
from ...errors import DimensionMismatchError
from ...regions import SimpleRegion
from ..linear_relaxations.alpha_resolvers import resolve_matmul_etas
from ..linear_relaxations.pairwise import compute_mul_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


def _unsqueeze_bounds(bounds: LinearBounds, *, axis: int) -> LinearBounds:
    """Return a copy of *bounds* with a size-1 dim inserted at *axis* of the bias.

    Linear coefficients carry trailing input axes after the bias prefix, so
    ``axis`` is resolved against the bias rank and only the bias-frame dim is
    inserted — the trailing input axes are preserved.
    """
    bias_ndim = bounds.bias_lower.ndim
    real_axis = axis if axis >= 0 else axis + bias_ndim + 1
    bias_lower = bounds.bias_lower.unsqueeze(real_axis)
    bias_upper = bounds.bias_upper.unsqueeze(real_axis)
    linear_lower = [lin.unsqueeze(real_axis) for lin in bounds.linear_lowers]
    linear_upper = [lin.unsqueeze(real_axis) for lin in bounds.linear_uppers]
    return LinearBounds(
        regions=bounds.regions,
        linear_lower=linear_lower or None,
        bias_lower=bias_lower,
        linear_upper=linear_upper or None,
        bias_upper=bias_upper,
        input_ids=list(bounds.input_ids) or None,
    )


def _squeeze_bounds(bounds: LinearBounds, *, axes: list[int]) -> LinearBounds:
    """Return a copy of *bounds* with the given bias-frame dims squeezed.

    Negative axes index from the right of the bias shape; the matching dim of
    every linear coefficient (which lives in the bias frame, not the input
    frame) is squeezed in step.
    """
    bias_lower = bounds.bias_lower
    bias_upper = bounds.bias_upper
    linear_lower = list(bounds.linear_lowers)
    linear_upper = list(bounds.linear_uppers)
    for axis in axes:
        bias_ndim = bias_lower.ndim
        real_axis = axis if axis >= 0 else axis + bias_ndim
        bias_lower = bias_lower.squeeze(real_axis)
        bias_upper = bias_upper.squeeze(real_axis)
        # Same bias-frame dim sits at the same absolute index on the linear
        # tensor (input axes come strictly after the bias prefix).
        linear_lower = [lin.squeeze(real_axis) for lin in linear_lower]
        linear_upper = [lin.squeeze(real_axis) for lin in linear_upper]
    return LinearBounds(
        regions=bounds.regions,
        linear_lower=linear_lower or None,
        bias_lower=bias_lower,
        linear_upper=linear_upper or None,
        bias_upper=bias_upper,
        input_ids=list(bounds.input_ids) or None,
    )


class ForwardLBPMatmul(ForwardLBPStrategy):
    """Forward LBP strategy for matmul (abstract@abstract, abstract@constant, constant@abstract)."""

    @staticmethod
    def _matmul_right_constant_linear(
        linear_lowers: list[torch.Tensor],
        linear_uppers: list[torch.Tensor],
        weight_pos: torch.Tensor,
        weight_neg: torch.Tensor,
        output_ndim: int,
        *,
        upper: bool,
    ) -> list[torch.Tensor]:
        transformed: list[torch.Tensor] = []
        for lower_linear, upper_linear in zip(linear_lowers, linear_uppers, strict=True):
            input_axes = lower_linear.shape[output_ndim:]
            batch_shape = lower_linear.shape[: output_ndim - 1]
            feature_dim = lower_linear.shape[output_ndim - 1]

            lower_flat = lower_linear.reshape(*batch_shape, feature_dim, -1)
            upper_flat = upper_linear.reshape(*batch_shape, feature_dim, -1)

            if upper:
                transformed_flat = torch.einsum("...kd,ko->...od", upper_flat, weight_pos) + torch.einsum(
                    "...kd,ko->...od", lower_flat, weight_neg
                )
            else:
                transformed_flat = torch.einsum("...kd,ko->...od", lower_flat, weight_pos) + torch.einsum(
                    "...kd,ko->...od", upper_flat, weight_neg
                )

            transformed.append(transformed_flat.reshape(*batch_shape, weight_pos.shape[1], *input_axes))

        return transformed

    @staticmethod
    def _matmul_left_constant_linear(
        linear_lowers: list[torch.Tensor],
        linear_uppers: list[torch.Tensor],
        weight_pos: torch.Tensor,
        weight_neg: torch.Tensor,
        output_ndim: int,
        *,
        upper: bool,
    ) -> list[torch.Tensor]:
        transformed: list[torch.Tensor] = []
        for lower_linear, upper_linear in zip(linear_lowers, linear_uppers, strict=True):
            input_axes = lower_linear.shape[output_ndim:]
            batch_shape = lower_linear.shape[: output_ndim - 1]
            feature_dim = lower_linear.shape[output_ndim - 1]

            lower_flat = lower_linear.reshape(*batch_shape, feature_dim, -1)
            upper_flat = upper_linear.reshape(*batch_shape, feature_dim, -1)

            if upper:
                transformed_flat = torch.einsum("ok,...kd->...od", weight_pos, upper_flat) + torch.einsum(
                    "ok,...kd->...od", weight_neg, lower_flat
                )
            else:
                transformed_flat = torch.einsum("ok,...kd->...od", weight_pos, lower_flat) + torch.einsum(
                    "ok,...kd->...od", weight_neg, upper_flat
                )

            transformed.append(transformed_flat.reshape(*batch_shape, weight_pos.shape[0], *input_axes))

        return transformed

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, LinearBounds) and isinstance(right, LinearBounds):
            return self._matmul_bounds_bounds_dispatch(left, right, node, ctx)

        if isinstance(left, LinearBounds) and isinstance(right, torch.Tensor):
            return self._matmul_right_constant(left, right)

        if isinstance(left, torch.Tensor) and isinstance(right, LinearBounds):
            return self._matmul_left_constant(left, right)

        raise TypeError(f"ForwardLBPMatmul requires at least one LinearBounds, got {type(left)} and {type(right)}")

    def _matmul_bounds_bounds_dispatch(
        self,
        a: LinearBounds,
        b: LinearBounds,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        """Handle the vector cases of ``torch.matmul`` by promoting to matrices.

        PyTorch documents five matmul rank-combinations:

        * 1-D @ 1-D — dot product, scalar output.
        * 2-D @ 1-D — promote ``b`` to ``(K, 1)``, matmul, squeeze trailing dim.
        * 1-D @ 2-D — promote ``a`` to ``(1, K)``, matmul, squeeze leading dim.
        * 2-D @ 2-D — direct.
        * N-D @ N-D — batched, possibly broadcasting leading dims.

        The two-abstract matrix path handles the last two natively; vector
        cases route through it by unsqueezing the vector operand(s) and
        squeezing the corresponding output dim.
        """
        a_vec = a.bias_lower.ndim == 1
        b_vec = b.bias_lower.ndim == 1
        if not (a_vec or b_vec):
            return self._matmul_bounds_bounds(a, b, node, ctx)

        a_promoted = _unsqueeze_bounds(a, axis=-2) if a_vec else a
        b_promoted = _unsqueeze_bounds(b, axis=-1) if b_vec else b
        out = self._matmul_bounds_bounds(a_promoted, b_promoted, node, ctx)

        squeeze_axes: list[int] = []
        if a_vec:
            squeeze_axes.append(-2)
        if b_vec:
            squeeze_axes.append(-1 if not a_vec else -1)
        # Squeeze trailing dim first to keep -2 valid for the second squeeze.
        return _squeeze_bounds(out, axes=squeeze_axes)

    def _matmul_right_constant(self, bounds: LinearBounds, weight: torch.Tensor) -> LinearBounds:
        """z = x @ W where x has linear bounds."""
        if weight.ndim != 2:
            raise DimensionMismatchError(f"matmul right operand must be 2D, got shape {tuple(weight.shape)}")

        if bounds.bias_lower.shape[-1] != weight.shape[0]:
            raise DimensionMismatchError(
                "matmul dimension mismatch: "
                f"bounds last dim {bounds.bias_lower.shape[-1]} vs "
                f"weight first dim {weight.shape[0]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim

        linear_lower = self._matmul_right_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=False,
        )

        bias_lower = torch.einsum("...k,ko->...o", bounds.bias_lower, weight_pos) + torch.einsum(
            "...k,ko->...o", bounds.bias_upper, weight_neg
        )

        linear_upper = self._matmul_right_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=True,
        )

        bias_upper = torch.einsum("...k,ko->...o", bounds.bias_upper, weight_pos) + torch.einsum(
            "...k,ko->...o", bounds.bias_lower, weight_neg
        )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )

    def _matmul_bounds_bounds(
        self,
        a: LinearBounds,
        b: LinearBounds,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        """McCormick relaxation for matmul ``z = a @ b`` with both operands abstract.

        Each output entry ``z[..., i, j] = sum_k a[..., i, k] * b[..., k, j]`` is
        a sum of independent bilinear terms. We apply the standard McCormick
        envelope to every ``a_ik * b_kj`` term, then reduce the per-term linear
        relaxation over ``k`` and compose with the linear bounds of ``a`` and
        ``b``.

        Requires both operands to be 2D or higher (matrix-matrix). Vector
        operands are not currently supported.
        """
        bounds_a = a.concretize()
        bounds_b = b.concretize()

        if bounds_a.lower.ndim < 2 or bounds_b.lower.ndim < 2:
            raise NotImplementedError(
                "Forward LBP matmul with two abstract operands requires each operand "
                f"to be at least 2D (matrix), got shapes {tuple(bounds_a.lower.shape)} "
                f"and {tuple(bounds_b.lower.shape)}."
            )

        try:
            batch_shape = torch.broadcast_shapes(
                bounds_a.lower.shape[:-2],
                bounds_b.lower.shape[:-2],
            )
        except RuntimeError as error:
            raise DimensionMismatchError(
                "matmul requires broadcastable batch dimensions, "
                f"got a.shape={tuple(bounds_a.lower.shape)} and b.shape={tuple(bounds_b.lower.shape)}"
            ) from error

        m_dim = bounds_a.lower.shape[-2]
        k_a = bounds_a.lower.shape[-1]
        k_b = bounds_b.lower.shape[-2]
        n_dim = bounds_b.lower.shape[-1]

        if k_a != k_b:
            raise DimensionMismatchError(f"matmul reduction dims mismatch: a.shape[-1]={k_a} vs b.shape[-2]={k_b}")

        # Broadcast the concrete bounds to the common batch shape, then shape
        # them as (*batch, M, K, 1) and (*batch, 1, K, N) so element-wise
        # broadcasting produces the full (*batch, M, K, N) bilinear-term grid.
        la = bounds_a.lower.expand(*batch_shape, m_dim, k_a).unsqueeze(-1)
        ua = bounds_a.upper.expand(*batch_shape, m_dim, k_a).unsqueeze(-1)
        lb = bounds_b.lower.expand(*batch_shape, k_a, n_dim).unsqueeze(-3)
        ub = bounds_b.upper.expand(*batch_shape, k_a, n_dim).unsqueeze(-3)

        # Reference for alpha-CROWN is the full (*batch, M, K, N) grid.
        reference = IntervalBounds(
            la.expand(*batch_shape, m_dim, k_a, n_dim),
            ua.expand(*batch_shape, m_dim, k_a, n_dim),
        )
        eta_lo, eta_up = resolve_matmul_etas(ctx.alpha_provider, node, reference)

        params = compute_mul_relaxation(
            IntervalBounds(la, ua),
            IntervalBounds(lb, ub),
            eta_lower=eta_lo if eta_lo is not None else 0.5,
            eta_upper=eta_up if eta_up is not None else 0.5,
        )

        # Sum the McCormick bias over K to land the initial bias at (*batch, M, N).
        bias_lower = params.bias_lower.sum(dim=-2)
        bias_upper = params.bias_upper.sum(dim=-2)

        al_pos = params.alpha_lower_a.clamp(min=0)
        al_neg = params.alpha_lower_a.clamp(max=0)
        au_pos = params.alpha_upper_a.clamp(min=0)
        au_neg = params.alpha_upper_a.clamp(max=0)
        bl_pos = params.alpha_lower_b.clamp(min=0)
        bl_neg = params.alpha_lower_b.clamp(max=0)
        bu_pos = params.alpha_upper_b.clamp(min=0)
        bu_neg = params.alpha_upper_b.clamp(max=0)

        # Bias contribution from a's affine: coefficient on bias_{lower,upper}_a.
        # bias_lower_a shape (*batch_a, M, K); unsqueeze(-1) broadcasts over N.
        a_bias_lower_bc = a.bias_lower.unsqueeze(-1)
        a_bias_upper_bc = a.bias_upper.unsqueeze(-1)
        bias_lower = bias_lower + (al_pos * a_bias_lower_bc + al_neg * a_bias_upper_bc).sum(dim=-2)
        bias_upper = bias_upper + (au_pos * a_bias_upper_bc + au_neg * a_bias_lower_bc).sum(dim=-2)

        # Bias contribution from b's affine: bias_lower_b shape (*batch_b, K, N);
        # unsqueeze(-3) introduces the M axis as a singleton.
        b_bias_lower_bc = b.bias_lower.unsqueeze(-3)
        b_bias_upper_bc = b.bias_upper.unsqueeze(-3)
        bias_lower = bias_lower + (bl_pos * b_bias_lower_bc + bl_neg * b_bias_upper_bc).sum(dim=-2)
        bias_upper = bias_upper + (bu_pos * b_bias_upper_bc + bu_neg * b_bias_lower_bc).sum(dim=-2)

        # Merge linear contributions by input_id so that shared regions accumulate.
        a_output_ndim = a.bias_lower.ndim  # (*batch_a, M, K) -> typically batch + 2
        b_output_ndim = b.bias_lower.ndim  # (*batch_b, K, N) -> typically batch + 2

        merged_lower: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        merged_upper: dict[int, tuple[SimpleRegion, torch.Tensor]] = {}
        ordered_ids: list[int] = []

        # a-contributions: linear term shape (*batch_a, M, K, *input_dims_a).
        # Insert N axis as singleton on the linear term; insert d axis as
        # singleton on the alpha params; then broadcast-multiply and sum over K.
        for iid, region, lin_low, lin_up in zip(a.input_ids, a.regions, a.linear_lowers, a.linear_uppers, strict=True):
            input_axes = lin_low.shape[a_output_ndim:]
            # lin shape -> (*batch_a, M, K, 1, *input_dims_a)
            lin_low_exp = lin_low.unsqueeze(a_output_ndim)
            lin_up_exp = lin_up.unsqueeze(a_output_ndim)

            # alpha shape -> (*batch, M, K, N, *1_for_input_dims)
            def _expand_alpha(alpha: torch.Tensor, extra: int) -> torch.Tensor:
                return alpha.reshape(alpha.shape + (1,) * extra)

            extra = len(input_axes)
            al_pos_e = _expand_alpha(al_pos, extra)
            al_neg_e = _expand_alpha(al_neg, extra)
            au_pos_e = _expand_alpha(au_pos, extra)
            au_neg_e = _expand_alpha(au_neg, extra)

            contrib_lower = (al_pos_e * lin_low_exp + al_neg_e * lin_up_exp).sum(dim=-2 - extra)
            contrib_upper = (au_pos_e * lin_up_exp + au_neg_e * lin_low_exp).sum(dim=-2 - extra)

            if iid in merged_lower:
                merged_lower[iid] = (merged_lower[iid][0], merged_lower[iid][1] + contrib_lower)
                merged_upper[iid] = (merged_upper[iid][0], merged_upper[iid][1] + contrib_upper)
            else:
                ordered_ids.append(iid)
                merged_lower[iid] = (region, contrib_lower)
                merged_upper[iid] = (region, contrib_upper)

        # b-contributions: linear term shape (*batch_b, K, N, *input_dims_b).
        # Insert M axis as singleton on the linear term; reduce over K.
        for iid, region, lin_low, lin_up in zip(b.input_ids, b.regions, b.linear_lowers, b.linear_uppers, strict=True):
            input_axes = lin_low.shape[b_output_ndim:]
            extra = len(input_axes)

            # lin shape (*batch_b, K, N, *input_dims_b) -> (*batch_b, 1, K, N, *input_dims_b)
            lin_low_exp = lin_low.unsqueeze(b_output_ndim - 2)
            lin_up_exp = lin_up.unsqueeze(b_output_ndim - 2)

            def _expand_alpha(alpha: torch.Tensor, extra: int) -> torch.Tensor:
                return alpha.reshape(alpha.shape + (1,) * extra)

            bl_pos_e = _expand_alpha(bl_pos, extra)
            bl_neg_e = _expand_alpha(bl_neg, extra)
            bu_pos_e = _expand_alpha(bu_pos, extra)
            bu_neg_e = _expand_alpha(bu_neg, extra)

            # Broadcast shape is (*batch, M, K, N, *input_dims); K lives at -2 - extra.
            contrib_lower = (bl_pos_e * lin_low_exp + bl_neg_e * lin_up_exp).sum(dim=-2 - extra)
            contrib_upper = (bu_pos_e * lin_up_exp + bu_neg_e * lin_low_exp).sum(dim=-2 - extra)

            if iid in merged_lower:
                merged_lower[iid] = (merged_lower[iid][0], merged_lower[iid][1] + contrib_lower)
                merged_upper[iid] = (merged_upper[iid][0], merged_upper[iid][1] + contrib_upper)
            else:
                ordered_ids.append(iid)
                merged_lower[iid] = (region, contrib_lower)
                merged_upper[iid] = (region, contrib_upper)

        regions = [merged_lower[iid][0] for iid in ordered_ids]
        linear_lower = [merged_lower[iid][1] for iid in ordered_ids]
        linear_upper = [merged_upper[iid][1] for iid in ordered_ids]

        return LinearBounds(
            regions=regions or None,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=ordered_ids or None,
        )

    def _matmul_left_constant(self, weight: torch.Tensor, bounds: LinearBounds) -> LinearBounds:
        """z = W @ x where x has linear bounds.

        PyTorch matmul semantics for ``W @ x`` distinguish the two operand-rank
        cases:

        * ``x`` is 1-D ``(K,)`` — the reduction is over ``K`` and the output is
          ``(M,)``.
        * ``x`` is N-D ``(*batch, K, N)`` for ``N >= 2`` — the reduction is over
          the second-to-last dim and the output is ``(*batch, M, N)``.

        Both cases share the same linear-bound math (positive / negative weight
        split, einsum on the linear coefficients) but differ in which abstract
        axis is the reduction axis.
        """
        if weight.ndim != 2:
            raise DimensionMismatchError(f"matmul left operand must be 2D, got shape {tuple(weight.shape)}")
        if bounds.bias_lower.ndim == 0:
            raise DimensionMismatchError("matmul abstract operand must be at least 1-D")

        if bounds.bias_lower.ndim == 1:
            return self._matmul_left_constant_vec(weight, bounds)
        return self._matmul_left_constant_mat(weight, bounds)

    def _matmul_left_constant_vec(self, weight: torch.Tensor, bounds: LinearBounds) -> LinearBounds:
        """W @ x with x 1-D ``(K,)`` → output ``(M,)``."""
        if bounds.bias_lower.shape[-1] != weight.shape[1]:
            raise DimensionMismatchError(
                "matmul dimension mismatch: "
                f"weight second dim {weight.shape[1]} vs "
                f"bounds last dim {bounds.bias_lower.shape[-1]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim

        linear_lower = self._matmul_left_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=False,
        )

        bias_lower = torch.einsum("ok,...k->...o", weight_pos, bounds.bias_lower) + torch.einsum(
            "ok,...k->...o", weight_neg, bounds.bias_upper
        )

        linear_upper = self._matmul_left_constant_linear(
            bounds.linear_lowers,
            bounds.linear_uppers,
            weight_pos,
            weight_neg,
            output_ndim,
            upper=True,
        )

        bias_upper = torch.einsum("ok,...k->...o", weight_pos, bounds.bias_upper) + torch.einsum(
            "ok,...k->...o", weight_neg, bounds.bias_lower
        )

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )

    def _matmul_left_constant_mat(self, weight: torch.Tensor, bounds: LinearBounds) -> LinearBounds:
        """W @ X with X N-D ``(*batch, K, N)``, ``N >= 2`` → output ``(*batch, M, N)``.

        The reduction axis is the second-to-last dim of ``X``; the trailing
        ``N`` dim is preserved on the output. Linear coefficients carry input
        axes after the bias prefix and are reshaped to fold ``N`` and the
        input axes into a single ``D`` for a 4-D einsum.
        """
        if bounds.bias_lower.shape[-2] != weight.shape[1]:
            raise DimensionMismatchError(
                "matmul dimension mismatch: "
                f"weight second dim {weight.shape[1]} vs "
                f"bounds second-to-last dim {bounds.bias_lower.shape[-2]}"
            )

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        output_ndim = bounds.bias_lower.ndim
        m_dim = weight.shape[0]
        k_dim = weight.shape[1]
        n_dim = bounds.bias_lower.shape[-1]

        bias_lower = torch.einsum("ok,...kn->...on", weight_pos, bounds.bias_lower) + torch.einsum(
            "ok,...kn->...on", weight_neg, bounds.bias_upper
        )
        bias_upper = torch.einsum("ok,...kn->...on", weight_pos, bounds.bias_upper) + torch.einsum(
            "ok,...kn->...on", weight_neg, bounds.bias_lower
        )

        def _transform(
            lower_linear: torch.Tensor,
            upper_linear: torch.Tensor,
            *,
            upper: bool,
        ) -> torch.Tensor:
            input_axes = lower_linear.shape[output_ndim:]
            batch_shape = lower_linear.shape[: output_ndim - 2]

            lower_flat = lower_linear.reshape(*batch_shape, k_dim, n_dim, -1)
            upper_flat = upper_linear.reshape(*batch_shape, k_dim, n_dim, -1)

            if upper:
                out = torch.einsum("ok,...knd->...ond", weight_pos, upper_flat) + torch.einsum(
                    "ok,...knd->...ond", weight_neg, lower_flat
                )
            else:
                out = torch.einsum("ok,...knd->...ond", weight_pos, lower_flat) + torch.einsum(
                    "ok,...knd->...ond", weight_neg, upper_flat
                )
            return out.reshape(*batch_shape, m_dim, n_dim, *input_axes)

        linear_lower = [
            _transform(low, up, upper=False) for low, up in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]
        linear_upper = [
            _transform(low, up, upper=True) for low, up in zip(bounds.linear_lowers, bounds.linear_uppers, strict=True)
        ]

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=bounds.input_ids,
        )
