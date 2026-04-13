# Copilot Instructions for bound_propagation

## Project Overview

This is a Python-based scientific computing project focused on verification of neural networks and beyond. The project is organized as a monorepo workspace using `uv` for dependency management.

## Software Engineering Design Practices

For design and architecture decisions, follow the practices in this section and any project-level contributing or design guidelines maintained in the repository.

Priority order for implementation decisions:
1. Mathematical Correctness
2. Pattern-driven extensibility (especially Strategy)
3. Testability and maintainability

### Pattern Guidance

- Prefer the **Strategy pattern** when behavior varies by algorithm, backend, or policy.
- Use **Factory/registry** wiring to select concrete strategy implementations.
- Use **Adapter pattern** to isolate framework-specific details behind shared interfaces.
- Prefer composition over deep inheritance.
- Keep domain modules dependent on abstractions, not backend-specific concrete implementations.

### Practical Rules for Contributors

- Avoid long `if/elif` trees for backend or method selection inside core logic; extract strategies instead.
- Avoid silent `try/except`-based fallbacks in core logic.
- Keep validation at API/module boundaries and error messages actionable.
- Do not introduce hidden fallbacks between strategies/backends.
- Keep public APIs typed and consistent with NumPy broadcast semantics where applicable.
- Use `from __future__ import annotations` for forward references in type hints to avoid string literals and enable better type checking.

## Research and Web Search Standards

When external knowledge is needed, prioritize fast and thorough research.

### Rapid Information Gathering

- Use `#websearch` for official documentation first.
- Use `#think` to analyze findings and plan implementation.
- Use `#websearch` for GitHub repositories and code examples.
- Use `#websearch` for Stack Overflow discussions and real-world issues.
- Use `#websearch` for performance benchmarks and comparisons.

### Source Priority Order

1. Official documentation (Python.org, library docs)
2. GitHub repositories with high stars/forks
3. Technical blogs from recognized experts
4. Stack Overflow with accepted answers
5. Academic papers for theoretical understanding

### Research Quality Standards

#### Information Validation

- Cross-reference findings across multiple sources.
- Check publication dates and prioritize recent information.
- Verify code examples work before implementing.
- Test assumptions with quick prototypes.

#### Performance Research

- Profile before optimizing; do not guess.
- Look for official benchmarking data.
- Check community feedback on performance.
- Consider real-world usage patterns, not just synthetic tests.

#### Dependency Evaluation

- Check maintenance status (last commit date, open issues).
- Review security vulnerability databases.
- Assess package size and import overhead.
- Verify license compatibility.

## Development Environment

### Prerequisites
- Python >= 3.11
- `uv` package manager for dependency management
- Project-local virtual environment at `.venv` (managed by `uv`)

### Mandatory Environment Rule
- Always use the project-local environment managed by `uv` for Python commands, tooling, tests, and scripts.
- Prefer `uv run <command>` for execution, `uvx <command>` for independent tools (e.g. ruff), and `uv sync` for environment synchronization.
- Do not run Python tooling from unrelated environments unless explicitly instructed.

### Setup
```bash

# Install uv if not already installed
pip install uv

# Install dependencies from declarative `pyproject.toml`
uv sync --group dev
```

### Workspace Management
This project uses `uv` workspaces. Each library in `libs/` is an independent package with its own `pyproject.toml`. When adding dependencies to a workspace member, use workspace references where appropriate:

```toml
[tool.uv.sources]
numeric-translators = { workspace = true }
```

## Code Style and Linting

### Linter
- **Ruff** is used for linting (version >= 0.14.10)
- Run linting with: `uv run ruff check .`
- Auto-fix issues with: `uv run ruff check --fix .`
- When using `plum.dispatch` overloads that redefine the same function name, add `# noqa: F811` on the redefined function declarations so Ruff does not flag the intentional override

### Python Style Guidelines
- Follow PEP 8 conventions, except use a modern maximum line length of 180 characters instead of the default PEP 8 limit
- Use type hints where appropriate (the project includes `py.typed` markers)
- Use docstrings for public APIs following NumPy-style documentation format
- All operations should adhere to NumPy broadcast semantics where applicable
- Use `snake_case` for variables and functions
- Use `CamelCase` for classes
- Single-letter variable names are only acceptable for loop indices (`i`, `j`, `k`)
- Avoid meaningless names such as `data`, `temp`, `stuff`; prefer descriptive names
- Avoid global variables; prefer explicit dependency injection and function/class scope

### Code Patterns
- Use abstract base classes (ABC) for defining interfaces
- Raise `NotImplementedError` for operations not supported in specific implementations
- Use `@dispatch` from plum-dispatch for multiple dispatch where needed
- Use context managers (`with` statements) for resource management; avoid manual cleanup patterns
- Prefer list comprehensions when they improve clarity over nested for-loop accumulation
- Prefer Python built-ins and stdlib helpers (`collections.Counter`, `itertools.chain`, `functools`) before custom re-implementations
- Avoid helper functions that only wrap a single library call.
    - helper functions are only usefull if they contain operation that are more than a couple of lines of code, or if they contain logic that is not specific to a single library.

### Error Handling Rules
- Use specific exceptions such as `ValueError`, `TypeError`, and domain-specific exceptions where appropriate
- Prefer adding domain-specific exception types (and typed exception fields when needed) over encoding excessive structured context into the exception message string
- Do not raise generic `Exception` in normal control flow
- Fail fast and fail loud: validate early and raise immediately with meaningful, actionable messages
- Prefer exceptions over return-code style error signaling

### Project Organization Rules
- Avoid flat "folder dump" layouts for new features
- Use clear package structure such as `utils/`, `models/`, and `tests/` where applicable
- Keep module boundaries cohesive and responsibility-driven

### Performance Rules
- Profile before optimizing using `cProfile` or `timeit`
- Do not introduce complexity for speculative micro-optimizations
- Use Pytorch performance tooling for anything involving pytorch and GPU optimization

## Testing

This repository includes tests across workspace packages. When adding or changing behavior:
- Add or update tests in the nearest package-level `tests/` directory
- Follow standard pytest conventions
- Prefer parameterized tests for cross-backend/shared-contract behavior
- Run focused tests first, then broader validation when needed
- Write unit tests with pytest for new logic and bug fixes
- If you add a new workspace package or move tests across packages, update the root `[tool.pytest.ini_options]` entries so discovery still matches the refactored layout

## Building and Documentation

### Documentation
- **Sphinx** is used for documentation (version >= 8.1.3)
- Sphinx is available as a dev dependency, but the repository does not currently maintain a checked-in `docs/` tree. Do not assume `docs/` and `docs/_build/` exist unless documentation sources are added as part of the change.

### Package Building
- Uses `uv_build` as the build backend (version >= 0.9.6, < 0.10.0)
- Build packages with: `uv build`

## Dependencies

### Adding Dependencies
When adding new dependencies:
1. Add them to the appropriate `pyproject.toml` file
2. For workspace members, use workspace references for internal dependencies
3. Specify version constraints appropriately
4. Run `uv sync` to update the lock file
5. Keep dependencies minimal; every added dependency must have clear value

### Development Dependencies
Development dependencies are specified in the root `pyproject.toml` under `[dependency-groups]`:
- `pytest` and `pytest-cov`: Test execution and coverage
- `ruff`: Linting and code formatting
- `sphinx`: Documentation generation
- Workspace extras for cross-package development, including Torch, JAX, Z3, Weights & Biases, and Marabou-related workflows where configured

## Git Workflow

- Follow conventional commit messages
- Keep changes focused and minimal
- Ensure code passes linting before committing
