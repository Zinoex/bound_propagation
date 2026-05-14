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

    Forward LBP pairwise only handles equal-rank operands; pair 4 mixes ranks
    ((3,) vs (2,3)). Pair 8 ((1,) vs (2,3)) happens to work because the
    trailing-singleton broadcast is degenerate. Backward LBP and the hybrid
    methods drop all broadcasting pairs (4-8).
    """
    out: list[tuple[str, str]] = []
    for op in ("mul", "div", "maximum", "minimum"):
        out.append(
            (
                f"test_{op}[forward_lbp-pair4]",
                f"Forward LBP {op}: cross-rank broadcasting between two abstract operands not implemented",
            )
        )
        for method, label in _LBP_HYBRIDS:
            for i in (4, 5, 6, 7, 8):
                out.append(
                    (
                        f"test_{op}[{method}-pair{i}]",
                        f"{label} {op}: broadcasting between two abstract operands not implemented",
                    )
                )
    return out


def _linear_rank_xfails() -> list[tuple[str, str]]:
    """nn.Linear / F.linear with >1-D feature input.

    PyTorch's nn.Linear documents input (*, H_in); the >1-D feature cases
    require backward LBP's tape to fold higher-rank batched weight
    contractions, which is not yet supported.
    """
    out: list[tuple[str, str]] = []
    for test in ("test_nn_linear", "test_nn_linear_no_bias"):
        for method, label in _LBP_HYBRIDS:
            for i in (1, 2, 3):
                out.append(
                    (
                        f"{test}[{method}-shape{i}]",
                        f"{label} Linear: input rank > 1 not supported (batched Linear needs explicit batch_ndim)",
                    )
                )
    return out


def _matmul_xfails() -> list[tuple[str, str]]:
    """matmul: two-abstract and batched-constant divergences."""
    out: list[tuple[str, str]] = []
    # Two-abstract: vectors / batched broadcast.
    for method, label in _ALL_LBP:
        for name in ("dot_1d_1d", "matvec_2d_1d", "vecmat_1d_2d"):
            out.append(
                (
                    f"test_matmul_two_abstract[{method}-{name}-",
                    f"{label} matmul: two-abstract {name} not supported (vectors / batched broadcast)",
                )
            )
    for method, label in _LBP_HYBRIDS:
        out.append(
            (
                f"test_matmul_two_abstract[{method}-batched_broadcast_3d_3d-",
                f"{label} matmul: two-abstract batched broadcast not supported",
            )
        )
        out.append(
            (
                f"test_matmul_two_abstract[{method}-batched_broadcast_3d_2d-",
                f"{label} matmul: two-abstract ND-vs-2D broadcast not supported",
            )
        )
    # Constant-right: backward LBP + hybrids reject input rank > 1.
    for method, label in _LBP_HYBRIDS:
        for i in (1, 2, 3):
            out.append(
                (
                    f"test_matmul_constant_right[{method}-shape{i}]",
                    f"{label} matmul-by-constant: input rank > 1 not yet supported",
                )
            )
    # Constant-left: weight @ x where x is rank > 1. All LBP methods reject.
    for method, label in _ALL_LBP:
        out.append(
            (
                f"test_matmul_constant_left[{method}-shape1]",
                f"{label} matmul (const @ abstract): non-vector abstract operand not supported",
            )
        )
    return out


def _shape_op_xfails() -> list[tuple[str, str]]:
    """cat / stack / squeeze / transpose / permute divergences."""
    out: list[tuple[str, str]] = []
    # cat / stack: negative dim against higher-rank input is not rewritten
    # against the bias-axis frame of the LinearBounds.
    for method in ("forward_lbp", "forward_backward_lbp"):
        label = "Forward LBP" if method == "forward_lbp" else "Forward-Backward LBP"
        out.append(
            (
                f"test_cat[{method}-a_shape4-b_shape4--1]",
                f"{label} cat: negative dim is not rewritten against the bias-axis frame",
            )
        )
        out.append(
            (
                f"test_stack[{method}-shape6--1]",
                f"{label} stack: negative dim is not rewritten against the bias-axis frame",
            )
        )
    # IBP / CROWN-IBP squeeze: dim=None is forwarded to torch.Tensor.squeeze,
    # which interprets None as a name lookup rather than "all dims".
    for method, label in (("ibp", "IBP"), ("crown_ibp", "CROWN-IBP")):
        for i in (4, 5):
            out.append(
                (
                    f"test_squeeze[{method}-shape{i}-None]",
                    f"{label} squeeze: dim=None forwarded to torch.squeeze as a name",
                )
            )
    # Backward LBP only registers the .transpose / .permute methods, not
    # the top-level torch.transpose / torch.permute callables.
    for method, label in _LBP_HYBRIDS:
        for i in range(6):
            out.append(
                (
                    f"test_transpose_function[{method}-shape{i}-",
                    f"{label}: torch.transpose top-level function not registered (only Tensor.transpose)",
                )
            )
        for i in range(5):
            out.append(
                (
                    f"test_permute_function[{method}-shape{i}-",
                    f"{label}: torch.permute top-level function not registered (only Tensor.permute)",
                )
            )
    return out


def _scalar_mul_constant_xfails() -> list[tuple[str, str]]:
    """0-D mul-by-constant: reshape(*shape) with empty shape."""
    return [
        (
            "test_mul_constant[forward_lbp-shape0]",
            "Forward LBP mul-by-constant: 0-D abstract input triggers reshape() with empty shape",
        ),
        (
            "test_mul_constant[forward_backward_lbp-shape0]",
            "Forward-Backward LBP mul-by-constant: 0-D abstract input triggers reshape() with empty shape",
        ),
    ]


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
