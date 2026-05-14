"""Pytest configuration for shape-compatibility tests.

Helpers live in :mod:`tests.test_compatibility._harness` so they can be
imported directly by the per-op test modules; this file also centralizes
the catalog of known cross-method shape divergences and marks them
``xfail`` so the suite stays green while the divergences are surfaced
in test reports.

Each entry is keyed by a substring that uniquely identifies a node-id
prefix; the value is the one-line reason. ``strict=True`` ensures the
xfail flips to a hard failure once an underlying bug is fixed, so the
catalog cannot drift.
"""

from __future__ import annotations

import pytest

_LBP_HYBRIDS = (
    ("backward_lbp", "Backward LBP"),
    ("forward_backward_lbp", "Forward-Backward LBP"),
    ("crown_ibp", "CROWN-IBP"),
)
_ALL_LBP = (("forward_lbp", "Forward LBP"), *_LBP_HYBRIDS)


def _broadcast_pairwise_xfails() -> list[tuple[str, str]]:
    """Pairwise broadcasting (mul / div / maximum / minimum) divergences.

    All fixed in this branch:

    * Forward LBP: ``PairedForwardRelaxation.forward`` aligns alpha and
      per-input linear coefficients when the output bias prefix is larger
      than an input's bias prefix.
    * Backward LBP (and hybrids that route through it): the pairwise
      strategies now broadcast both operand interval bounds to the common
      output shape before computing the McCormick / min-max relaxation,
      store each operand's pre-broadcast shape on the relaxation, and the
      ``backward_through`` method sum-reduces the per-operand ``A``
      contributions back to the operand's natural shape.
    """
    return []


def _linear_rank_xfails() -> list[tuple[str, str]]:
    """nn.Linear / F.linear with >1-D feature input.

    Fixed in this branch: the backward LBP ``LinearBackwardRelaxation``
    routes the Identity fast path to the generic dense dispatch when the
    operator's feature_shape carries extra leading dims (the Kronecker
    ``I_leading ⊗ W`` structure), and the bias contribution is now
    computed via ``A.apply(broadcast(bias))`` semantics so the resulting
    bias shape matches the operator's output shape regardless of rank.
    """
    return []


def _matmul_xfails() -> list[tuple[str, str]]:
    """matmul: two-abstract and batched-constant divergences.

    Fixed in this branch:

    * ``MatmulBothAbstractRelaxation`` now promotes 1-D vector operands to
      matrices at build time (mirroring PyTorch's matmul semantics), then
      unsqueezes the upstream ``A`` to match the promoted matmul output
      shape and squeezes the promoted axes back out of each per-operand
      ``A``-term.
    * The same relaxation carries the un-promoted operand shapes and
      sum-reduces per-operand ``A``-terms across any batched-broadcast
      dims, so ND-vs-ND and ND-vs-2-D matmul work end-to-end.
    * ``MatmulRightConstantRelaxation`` no longer leaks a node-axis into
      the bias contribution; the zero-bias is built from
      ``A.output_shape`` directly so any input rank is supported.
    * ``MatmulLeftConstantRelaxation`` now contracts ``A``'s "M" axis (the
      first input axis, not the trailing axis) against ``W``'s first dim,
      so non-vector abstract operands work end-to-end.
    """
    return []


def _shape_op_xfails() -> list[tuple[str, str]]:
    """cat / stack / squeeze / transpose / permute divergences.

    Fixed in this branch:

    * Forward LBP cat / stack with negative ``dim`` are now rewritten against
      the bias-axis frame so they accept the same shapes PyTorch documents.
    * IBP / CROWN-IBP ``squeeze`` now handles ``dim=None`` (squeeze all
      size-1 dims) without forwarding it as a name.
    * Backward LBP registry now binds top-level ``torch.transpose`` and
      ``torch.permute`` in addition to the ``Tensor`` methods.
    """
    return []


def _scalar_mul_constant_xfails() -> list[tuple[str, str]]:
    """0-D mul-by-constant.

    Fixed: ForwardLBPMul._multiply_by_constant now builds the broadcast
    shape as a single tuple so the reshape call is well-formed when both
    the constant and the linear coefficient are 0-D.
    """
    return []


# Materialize the full catalog. Each entry is (nodeid-substring, reason); a
# test is marked xfail iff its nodeid contains the substring.
KNOWN_XFAILS: list[tuple[str, str]] = [
    *_broadcast_pairwise_xfails(),
    *_scalar_mul_constant_xfails(),
    *_linear_rank_xfails(),
    *_matmul_xfails(),
    *_shape_op_xfails(),
]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply xfail markers to known cross-method shape divergences.

    Each entry's substring is matched against ``item.nodeid``; matching
    items get a strict xfail marker so the catalog stays in sync with
    the actual behavior.
    """
    del config
    for item in items:
        if "test_compatibility" not in item.nodeid:
            continue
        for substring, reason in KNOWN_XFAILS:
            if substring in item.nodeid:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
                break
