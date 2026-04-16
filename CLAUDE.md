# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
uv sync --group dev
```

### Testing
Always use `uv run pytest` — never bare `pytest`:
```bash
uv run pytest
uv run pytest -k <pattern>
uv run pytest tests/test_forward_lbp/
```

### Linting and Formatting
Always use `uvx ruff` — never bare `ruff`:
```bash
uvx ruff check .
uvx ruff check --fix .
uvx ruff format .
uvx ruff format --check .
```

### Type Checking
Always use `uvx ty check` — never bare `ty`:
```bash
uvx ty check src/
uvx ty check src/bound_propagation/propagation/
```

### Build
```bash
uv build
```

## Architecture

**bound_propagation** is a PyTorch library for neural network verification via bound propagation. It traces a neural network's computation graph using `torch.fx` and propagates bounds (interval or linear) through it.

### Core Abstractions

**Bounds** (`src/bound_propagation/bounds/`): Two bound representations:
- `IntervalBounds`: Simple `[lower, upper]` interval tensors
- `LinearBounds`: Affine relaxations `W_lower @ x + b_lower ≤ y ≤ W_upper @ x + b_upper`

**Regions** (`src/bound_propagation/regions/`): Describe input spaces (e.g., `HyperRectangle` — a box constraint on inputs).

**Tracer** (`src/bound_propagation/tracer/`): `BoundPropagationTracer` extends `torch.fx.Tracer` to convert an `nn.Module` into an `fx.Graph` for analysis. Raises domain exceptions (`TraceError`, `UnsupportedOperationError`) on unsupported operations.

**Propagation** (`src/bound_propagation/propagation/`): The core engine:
- `PropagationContext`: Holds the bounds store (node name → computed bounds) and resolves arguments during graph traversal.
- `TargetRegistry`: Maps `torch.fx` node targets (callables, `nn.Module` types) to bounding strategy implementations — this is the Factory pattern.
- `BoundPropagator` (ABC, `propagation/methods/base.py`): Abstract propagator that walks the graph and dispatches to strategies.
  - `IBPPropagator`: Interval Bound Propagation — fast, less precise.
  - `ForwardLBPPropagator`: Forward linear bound propagation.
  - Backward LBP (CROWN): Handled via `BackwardBoundingStrategy` implementations.

**Strategies** (`propagation/ibp/`, `propagation/forward_lbp/`, `propagation/backward_lbp/`): One file per operation (relu, sigmoid, log, etc.), each implementing the relevant strategy interface. `propagation/linear_relaxations/` provides shared relaxation math reused across forward and backward strategies.

### Design Principles

Priority order: **Mathematical correctness → Pattern-driven extensibility → Testability**.

- **Strategy pattern** for per-operation bounding behavior; avoid `if/elif` trees for method selection.
- **Factory/registry** (`TargetRegistry`) to wire targets to strategies.
- **Composition over inheritance**; keep domain modules depending on abstractions.
- Use `@dispatch` from `plum-dispatch` for multiple dispatch; add `# noqa: F811` on redefined overloads.
- Use `from __future__ import annotations` for forward references in type hints.

### Imports

Within `src/bound_propagation`, always use **relative imports**. Import from the **highest-level re-export** available in `__init__.py` files, unless you are inside the same submodule that defines the symbol (then import from the local sibling directly).

```python
# GOOD
from ...bounds import IntervalBounds

# BAD
from bound_propagation.bounds.interval_bounds import IntervalBounds
```

### Error Handling

Fail early and loud:
- Validate inputs at public API entry points before any computation.
- Raise immediately with `ValueError`, `TypeError`, or `NotImplementedError`; never return sentinels.
- Never use `assert` for runtime validation (disabled with `-O`).
- Never swallow exceptions silently; chain with `from e` when wrapping.
- Error messages must include **what** failed, **why**, and **what value was seen**.

### Code Style

- 120-character line length (not PEP 8's 79).
- Double quotes (enforced by ruff).
- NumPy-style docstrings; no trailing whitespace on blank lines inside docstrings (`ruff format` does **not** fix this — check manually).
- Single-letter names only acceptable for loop indices (`i`, `j`, `k`).
- Avoid helper functions that only wrap a single library call.
