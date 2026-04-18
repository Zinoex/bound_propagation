# Backward LBP: Wengert List Refactor — Implementation Plan

## Context

Replace the recursive symbolic tree approach for backward LBP (CROWN) with a Wengert list (tape-based) approach. The current module is non-functional scaffolding (`linear_relaxations/base.py` doesn't exist, propagator is commented out). The math in the existing `Symbolic*` classes is correct and serves as reference.

Design decisions from discussion:
- Interleaved concretization (recursive CROWN on partial tape, not IBP)
- Bias threading through returns (caller accumulates)
- Caching concretized bounds in the tape
- `BackwardRelaxation` with `predecessor_nodes()` + `backward_through()` interface
- Subgraph + pending counter computed from relaxation predecessors (not fx.Node args)
- Internal accumulation in `a_terms` when same predecessor appears multiple times

## Execution Strategy

This refactor has **sequential dependencies** that prevent fully parallel execution:

```
Phase 1: Core infrastructure (coordinator implements directly, commits to branch)
    ↓
Phase 2: Operation-specific relaxations (5 parallel workers in worktrees)
    ↓
Phase 3: Integration + tests (coordinator implements directly)
```

## Unit 1: Core Infrastructure (coordinator, sequential)

**Files:** `backward_lbp/base.py`, `backward_lbp/tape.py`, `methods/backward_lbp_propagator.py`

### `backward_lbp/base.py`

```python
@dataclass
class BackwardContributions:
    a_terms: dict[fx.Node, tuple[Tensor, Tensor]]  # pred -> (delta_A_lower, delta_A_upper)
    bias_lower: Tensor
    bias_upper: Tensor

class BackwardRelaxation(ABC):
    @abstractmethod
    def predecessor_nodes(self) -> list[fx.Node]:
        """Unique fx.Node predecessors this relaxation propagates A-matrices to."""

    @abstractmethod
    def backward_through(self, A_lower: Tensor, A_upper: Tensor, batch_ndim: int) -> BackwardContributions:
        """Single-step backward. Returns contributions to predecessors + bias deltas."""

class IntervalLeafRelaxation(BackwardRelaxation):
    """Chain-breaking leaf: contributes only to bias, no predecessors."""
    lower: Tensor
    upper: Tensor
    # predecessor_nodes() -> []
    # backward_through: sign decomposition on A, contract over node dims into bias

class BackwardLBPStrategy(BoundingStrategy):
    @abstractmethod
    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> BackwardRelaxation:
        ...
```

### `backward_lbp/tape.py`

```python
class BackwardTape:
    def __init__(self, graph_module, input_regions): ...

    # Storage
    def record(self, node, relaxation): ...
    def record_concrete(self, node, value): ...
    def resolve(self, arg): ...
    def resolve_args(self, node): ...

    # Backward algorithm
    def backward_from(self, node, batch_ndim) -> LinearBounds:
        """Run backward LBP from node to inputs."""
        subgraph = self._backward_subgraph(node)  # follows relaxation.predecessor_nodes()
        pending = self._compute_pending(subgraph)  # counts from relaxation predecessors
        # BFS with A-matrix accumulation + bias threading
        # Collect accumulated A at placeholders -> LinearBounds

    def concretize_at(self, node, batch_ndim) -> IntervalBounds:
        """backward_from + concretize, with cache."""
        if node in self._interval_cache: return cached
        result = self.backward_from(node, batch_ndim).concretize()
        self._interval_cache[node] = result
        return result

    # Subgraph helpers
    def _backward_subgraph(self, start) -> set[fx.Node]:
        """BFS backward following relaxation.predecessor_nodes(), not fx.Node.args."""

    def _compute_pending(self, subgraph) -> dict[fx.Node, int]:
        """Count unique predecessors per relaxation within subgraph."""
```

Key design: subgraph + pending use `relaxation.predecessor_nodes()` to avoid deadlocks from chain-breaking ops (IntervalLeafRelaxation has no predecessors but fx.Node still has args).

### `methods/backward_lbp_propagator.py`

Two-phase propagator:
1. Forward: seed placeholders, build tape via strategies
2. Final: `tape.backward_from(output_node, batch_ndim=0)`

## Units 2-6: Operation-Specific Relaxations (parallel workers)

Each worker rewrites one `backward_lbp/*.py` file. Each worker receives the complete `BackwardRelaxation` interface definition and the math reference from the existing `Symbolic*` classes.

### Unit 2: Linear Operations (`backward_lbp/linear.py`)

Relaxation classes:
- `LinearBackwardRelaxation(weight, bias, input_node)` — y = x @ W^T + b
- `MatmulRightConstantRelaxation(weight, input_node)` — y = x @ W
- `MatmulLeftConstantRelaxation(weight, input_node)` — y = W @ x
- `AddRelaxation(left_node, right_node)` — y = x1 + x2 (both abstract)
- `SubRelaxation(left_node, right_node)` — y = x1 - x2
- `ConstantAddRelaxation(constant, input_node)` — y = x + c
- `NegRelaxation(input_node)` — y = -x
- `ScaleRelaxation(scale, input_node)` — y = c * x

All are purely linear transforms on A-matrices. No sign decomposition needed.
`predecessor_nodes()` returns `[input_node]` for unary, `list({left_node, right_node})` for binary.

### Unit 3: Elementwise Operations (`backward_lbp/elementwise.py`)

Single relaxation class:
- `ElementwiseBackwardRelaxation(params: ElementwiseParams, input_node)` — sign decomposition

Strategies use `tape.concretize_at()` to get interval bounds, then `compute_*_relaxation()` from `linear_relaxations/elementwise.py`.

### Unit 4: Pairwise Operations (`backward_lbp/pairwise.py`)

Single relaxation class:
- `PairedBackwardRelaxation(params: PairedParams, left_node, right_node)` — sign decomposition on two coefficient sets

**Must handle `left_node == right_node`** (e.g., `x * x`): use `_accumulate()` helper when building a_terms.

Strategies use `tape.concretize_at()` for both inputs.
Special cases: abstract*constant → `ScaleRelaxation`, abstract/constant → `ScaleRelaxation(1/c)`.

### Unit 5: Shape Operations (`backward_lbp/shape.py`)

Relaxation classes:
- `ReshapeRelaxation`, `UnsqueezeRelaxation`, `SqueezeRelaxation`
- `TransposeRelaxation`, `PermuteRelaxation`
- `SelectRelaxation`, `GetItemRelaxation`
- `CatRelaxation(input_nodes: list[fx.Node], ...)`, `StackRelaxation(input_nodes, ...)`

Pure A-matrix dimension transforms. No sign decomposition. No `tape.concretize_at()`.
Cat/Stack: `predecessor_nodes()` returns `list(set(self.input_nodes))`. `backward_through` uses `_accumulate()` when splitting A-matrices to multiple inputs.

### Unit 6: Reduction Operations (`backward_lbp/reduction.py`)

Relaxation classes:
- `SumRelaxation(dim, keepdim, source_shape, input_node)` — expand A-matrices
- `MeanRelaxation` — delegates to SumRelaxation with A/count

Strategies for amax/amin: use `tape.concretize_at()`, apply reduction, return `IntervalLeafRelaxation`.

## Unit 7: Integration + Tests (coordinator, sequential)

**Files:** `backward_lbp/__init__.py`, `methods/__init__.py`, `propagation/__init__.py`, tests

### Registry (`backward_lbp/__init__.py`)

`create_default_backward_lbp_registry()` wiring all strategies to targets.

### Exports

Uncomment `BackwardLBPPropagator` in `methods/__init__.py` and `propagation/__init__.py`.

### Cleanup

- Delete `backward_lbp/utils.py` (`_merge_backward_bounds` is no longer needed)
- Remove broken imports from non-existent `linear_relaxations/base.py`, `linear_relaxations/linear.py`, etc.

### Tests (`tests/test_backward_lbp/`)

**Existing tests** (`test_full_workflow.py`): update imports, verify they pass.

**New tests asserting on LinearBounds directly** (not just concretized intervals):

**Linear operations (exact — linear_lower == linear_upper, bias_lower == bias_upper):**
- Identity: `linear == I, bias == 0`
- `y = Wx + b`: `linear == W, bias == b`
- Chain `y = W2(W1 x + b1) + b2`: `linear == W2 @ W1, bias == W2 @ b1 + b2`
- Fan-out `y = x + x`: `linear == 2I, bias == 0`
- Negation: `linear == -I, bias == 0`
- Scale `y = 3x`: `linear == 3I, bias == 0`

**Nonlinear regime-specific:**
- ReLU positive (`x in [1,3]`): exact identity → `linear == I, bias == 0`
- ReLU negative (`x in [-3,-1]`): exact zero → `linear == 0, bias == 0` (no linear terms)
- ReLU crossing (`x in [-2,3]`): `linear_upper == 0.6, bias_upper == 1.2, linear_lower == 0.6, bias_lower == 0`

**Edge cases:**
- `y = x + x` — fan-out accumulation in backward
- `y = x * x` — same node as both pairwise inputs, internal accumulation
- Diamond: `a = relu(x); b = sigmoid(x); y = a + b` — accumulation from two chains
- Chain-breaking: `y = amax(relu(x))` — IntervalLeafRelaxation, no backward through amax
- Multi-input: `f(x1, x2) = x1 @ W + x2` — LinearBounds with 2 regions/input_ids
- Zero-width region: `x in [2, 2]` — degenerate interval, exact point bounds
- Scalar: single-element input/output
- Deep chain: 3+ layers with nonlinearities, verify soundness
- CROWN >= Forward LBP tightness for fan-out networks

## Verification

```bash
uvx ruff check src/bound_propagation/propagation/backward_lbp/
uvx ruff format --check src/bound_propagation/propagation/backward_lbp/
uv run pytest tests/test_backward_lbp/ -v
uv run pytest  # full suite, no regressions
```

## Conventions

- Relative imports within `src/bound_propagation`
- `from __future__ import annotations` in all files
- Double quotes, 120-char line length
- `@final @dataclass` for concrete relaxation classes
- NumPy-style docstrings, no trailing whitespace on blank docstring lines
- No `assert` for runtime validation
- Use `@dispatch` from `plum-dispatch` for multiple dispatch; `# noqa: F811` on redefined overloads
- Error messages: what failed, why, what value was seen
