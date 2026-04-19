"""Graph simplification pass for ``torch.fx`` graphs.

Rewrites sub-DAGs into simpler equivalent forms.  Fewer non-linearities
or shorter compositions generally yield tighter bound-propagation
relaxations.

The pass runs a fixed-point loop over a list of pure rewriters, each
detecting a single pattern on a node and mutating the graph in place.
After saturation, dead code is eliminated and the module is recompiled.

Shape-dependent rewrites (no-op ``reshape``/``view``) consult
``node.meta["tensor_meta"]``; run :class:`MetadataPass` first if you
want them to fire.  Rewrites invalidate meta on changed nodes; re-run
:class:`MetadataPass` afterwards if downstream passes need it.

Future work (not in this first cut): trigonometric product identities,
``exp(x) * exp(y) → exp(x + y)``, linear factoring gated on
``is_abstract`` metadata, and constant folding of non-abstract sub-DAGs.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TypeGuard

import torch
import torch.fx as fx

Rewriter = Callable[[fx.GraphModule, fx.Node], bool]


# ---------------------------------------------------------------------------
# Target sets.  Each covers the function, tensor-method, and call_method
# (string) forms that ``torch.fx`` can produce for the same operation.
# ---------------------------------------------------------------------------

_ADD_TARGETS: set[object] = {torch.add, operator.add, torch.Tensor.add, "add"}
_SUB_TARGETS: set[object] = {torch.sub, operator.sub, torch.Tensor.sub, "sub"}
_MUL_TARGETS: set[object] = {torch.mul, operator.mul, torch.Tensor.mul, "mul"}
_DIV_TARGETS: set[object] = {torch.div, operator.truediv, torch.Tensor.div, "div"}
_NEG_TARGETS: set[object] = {torch.neg, operator.neg, torch.Tensor.neg, "neg"}
_EXP_TARGETS: set[object] = {torch.exp, torch.Tensor.exp, "exp"}
_LOG_TARGETS: set[object] = {torch.log, torch.Tensor.log, "log"}

_RESHAPE_TARGETS: set[object] = {torch.reshape, torch.Tensor.reshape, "reshape"}
_VIEW_TARGETS: set[object] = {torch.Tensor.view, "view"}
_SQUEEZE_TARGETS: set[object] = {torch.squeeze, torch.Tensor.squeeze, "squeeze"}
_UNSQUEEZE_TARGETS: set[object] = {torch.unsqueeze, torch.Tensor.unsqueeze, "unsqueeze"}
_TRANSPOSE_TARGETS: set[object] = {torch.transpose, torch.Tensor.transpose, "transpose"}


# ---------------------------------------------------------------------------
# Matcher helpers
# ---------------------------------------------------------------------------


def _is_op(node: object, targets: set[object]) -> TypeGuard[fx.Node]:
    if not isinstance(node, fx.Node):
        return False
    if node.op not in ("call_function", "call_method"):
        return False
    return node.target in targets


def _is_literal_scalar(x: object, value: float | None = None) -> TypeGuard[int | float]:
    # ``bool`` is excluded: ``isinstance(True, int)`` is True in Python,
    # but boolean literals are semantically distinct from the ``0``/``1``
    # we want to match.
    if isinstance(x, fx.Node) or isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    if value is None:
        return True
    return float(x) == float(value)


def _replace(gm: fx.GraphModule, old: fx.Node, new: fx.Node) -> None:
    old.replace_all_uses_with(new)
    gm.graph.erase_node(old)


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------


def _rewrite_add_zero(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x + 0 → x`` and ``0 + x → x``."""
    if not _is_op(node, _ADD_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if isinstance(a, fx.Node) and _is_literal_scalar(b, 0):
        _replace(gm, node, a)
        return True
    if isinstance(b, fx.Node) and _is_literal_scalar(a, 0):
        _replace(gm, node, b)
        return True
    return False


def _rewrite_sub_zero(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x - 0 → x``."""
    if not _is_op(node, _SUB_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if isinstance(a, fx.Node) and _is_literal_scalar(b, 0):
        _replace(gm, node, a)
        return True
    return False


def _rewrite_mul_one(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x * 1 → x`` and ``1 * x → x``."""
    if not _is_op(node, _MUL_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if isinstance(a, fx.Node) and _is_literal_scalar(b, 1):
        _replace(gm, node, a)
        return True
    if isinstance(b, fx.Node) and _is_literal_scalar(a, 1):
        _replace(gm, node, b)
        return True
    return False


def _rewrite_div_one(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x / 1 → x``."""
    if not _is_op(node, _DIV_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if isinstance(a, fx.Node) and _is_literal_scalar(b, 1):
        _replace(gm, node, a)
        return True
    return False


def _rewrite_div_by_const(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x / c → x * (1/c)`` for constant ``c ∉ {0, 1}``.

    Turns a pairwise division into a scalar multiplication, which bound-
    propagation handles more tightly because it is linear in ``x``.
    """
    if not _is_op(node, _DIV_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if not isinstance(a, fx.Node):
        return False
    if not _is_literal_scalar(b) or float(b) == 0.0 or float(b) == 1.0:
        return False
    factor = 1.0 / float(b)
    with gm.graph.inserting_before(node):
        new_node = gm.graph.call_function(operator.mul, (a, factor))
    _replace(gm, node, new_node)
    return True


def _rewrite_double_neg(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``-(-x) → x``."""
    if not _is_op(node, _NEG_TARGETS) or len(node.args) != 1:
        return False
    inner = node.args[0]
    if not _is_op(inner, _NEG_TARGETS) or len(inner.args) != 1:
        return False
    x = inner.args[0]
    if not isinstance(x, fx.Node):
        return False
    _replace(gm, node, x)
    return True


# ---------------------------------------------------------------------------
# Self-product → power
# ---------------------------------------------------------------------------


def _rewrite_self_mul(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``x * x → pow(x, 2)``.

    The product relaxation of ``x * x`` loses the ``a == b`` coupling
    and is strictly looser than a dedicated square relaxation whenever
    ``x`` straddles zero.
    """
    if not _is_op(node, _MUL_TARGETS) or len(node.args) != 2:
        return False
    a, b = node.args
    if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
        return False
    if a is not b:
        return False
    with gm.graph.inserting_before(node):
        new_node = gm.graph.call_function(torch.pow, (a, 2))
    _replace(gm, node, new_node)
    return True


# ---------------------------------------------------------------------------
# Exp / log identities
# ---------------------------------------------------------------------------


def _rewrite_log_of_exp(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``log(exp(x)) → x``.

    Valid unconditionally: ``exp`` maps into ``(0, ∞)`` where ``log`` is
    the exact inverse.  Removes two non-linearities in one shot.
    """
    if not _is_op(node, _LOG_TARGETS) or len(node.args) != 1:
        return False
    inner = node.args[0]
    if not _is_op(inner, _EXP_TARGETS) or len(inner.args) != 1:
        return False
    x = inner.args[0]
    if not isinstance(x, fx.Node):
        return False
    _replace(gm, node, x)
    return True


# ---------------------------------------------------------------------------
# Structural no-ops
# ---------------------------------------------------------------------------


def _tensor_shape(node: fx.Node) -> tuple[int, ...] | None:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    return meta.get("shape")


def _rewrite_noop_reshape(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``reshape(x, shape_of_x) → x`` (and ``view`` likewise).

    Fires only when both input and output shapes are known via
    ``node.meta["tensor_meta"]``; without :class:`MetadataPass` this
    rewriter is a no-op.
    """
    if not _is_op(node, _RESHAPE_TARGETS | _VIEW_TARGETS):
        return False
    if not node.args or not isinstance(node.args[0], fx.Node):
        return False
    x = node.args[0]
    x_shape = _tensor_shape(x)
    out_shape = _tensor_shape(node)
    if x_shape is None or out_shape is None:
        return False
    if x_shape != out_shape:
        return False
    _replace(gm, node, x)
    return True


def _rewrite_squeeze_unsqueeze(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``squeeze(unsqueeze(x, d), d) → x``.

    Only fires when both calls carry an explicit, equal ``dim`` literal;
    the dim-less ``squeeze`` form has data-dependent behaviour we can't
    simplify statically.
    """
    if not _is_op(node, _SQUEEZE_TARGETS) or len(node.args) != 2:
        return False
    inner = node.args[0]
    if not _is_op(inner, _UNSQUEEZE_TARGETS) or len(inner.args) != 2:
        return False
    if node.args[1] != inner.args[1]:
        return False
    x = inner.args[0]
    if not isinstance(x, fx.Node):
        return False
    _replace(gm, node, x)
    return True


def _rewrite_double_transpose(gm: fx.GraphModule, node: fx.Node) -> bool:
    """``transpose(transpose(x, a, b), a, b) → x``.

    ``transpose`` is symmetric in its two dim args, so ``{a, b}`` equality
    is the right comparison.
    """
    if not _is_op(node, _TRANSPOSE_TARGETS) or len(node.args) != 3:
        return False
    inner = node.args[0]
    if not _is_op(inner, _TRANSPOSE_TARGETS) or len(inner.args) != 3:
        return False
    if {node.args[1], node.args[2]} != {inner.args[1], inner.args[2]}:
        return False
    x = inner.args[0]
    if not isinstance(x, fx.Node):
        return False
    _replace(gm, node, x)
    return True


# ---------------------------------------------------------------------------
# Pass driver
# ---------------------------------------------------------------------------


def default_rewriters() -> list[Rewriter]:
    """Return the default rewriter list in a safe order.

    The ``div`` rewrites are ordered so ``x / 1`` wins before ``x / c``.
    """
    return [
        _rewrite_add_zero,
        _rewrite_sub_zero,
        _rewrite_mul_one,
        _rewrite_div_one,
        _rewrite_double_neg,
        _rewrite_self_mul,
        _rewrite_log_of_exp,
        _rewrite_noop_reshape,
        _rewrite_squeeze_unsqueeze,
        _rewrite_double_transpose,
        _rewrite_div_by_const,
    ]


class SimplificationPass:
    """Rewrite a traced graph into a simpler equivalent form.

    Runs each rewriter over every node in a fixed-point loop: eliminating
    one pattern can expose another (e.g. ``log(exp(x * 1)) → log(exp(x)) → x``).
    After saturation, :func:`torch.fx.Graph.eliminate_dead_code` prunes
    unreferenced nodes and the graph is recompiled.

    Parameters
    ----------
    rewriters : list of callables, optional
        Each callable takes ``(graph_module, node)`` and returns ``True``
        if it rewrote the graph.  Defaults to :func:`default_rewriters`.
    max_iterations : int, optional
        Safety cap on outer loop iterations.  Raises :class:`RuntimeError`
        if exceeded, indicating a rewriter pair is oscillating.

    Example
    -------
    >>> gm = BoundPropagationTracer(registry).trace(model)
    >>> MetadataPass(gm).run(example, abstract_mask=[True])
    >>> SimplificationPass().run(gm)
    """

    def __init__(
        self,
        rewriters: list[Rewriter] | None = None,
        max_iterations: int = 100,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        self.rewriters: list[Rewriter] = list(rewriters) if rewriters is not None else default_rewriters()
        self.max_iterations = max_iterations

    def run(self, graph_module: fx.GraphModule) -> fx.GraphModule:
        """Rewrite ``graph_module`` in place and return it."""
        for _ in range(self.max_iterations):
            if not self._one_pass(graph_module):
                graph_module.graph.eliminate_dead_code()
                graph_module.graph.lint()
                graph_module.recompile()
                return graph_module

        raise RuntimeError(
            f"SimplificationPass did not converge within {self.max_iterations} iterations; "
            "suspect an oscillating pair of rewriters."
        )

    def _one_pass(self, graph_module: fx.GraphModule) -> bool:
        """One sweep over every live node; returns True if any rewrite fired."""
        changed = False
        for node in list(graph_module.graph.nodes):
            # ``replace_all_uses_with`` + ``erase_node`` detaches a node
            # from its graph; skip such stale references.
            if node.graph is None:
                continue
            for rewriter in self.rewriters:
                if rewriter(graph_module, node):
                    changed = True
                    break
        return changed
