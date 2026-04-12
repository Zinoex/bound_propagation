---
description: "Use when running or suggesting linting or formatting commands in this project. Enforces Ruff execution through uvx."
---
# Lint and Format Command Rules

## Rule

Use `uvx ruff` for all Ruff lint and format invocations in this repository.

Do not use bare `ruff` commands.

## Canonical Commands

```bash
# Lint check
uvx ruff check .

# Lint auto-fix
uvx ruff check --fix .

# Format
uvx ruff format .

# Format check (no edits)
uvx ruff format --check .
```

## Examples

```bash
# GOOD
uvx ruff check .
uvx ruff check --fix .
uvx ruff format .

# BAD
ruff check .
ruff format .
```

## Notes

- Keep normal Ruff flags and paths; only the command prefix is enforced.
- This aligns with the project's tooling in `DEVELOPMENT.md`.
