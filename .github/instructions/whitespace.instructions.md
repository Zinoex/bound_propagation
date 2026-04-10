---
description: "Use when writing or reviewing Python code. Empty lines must not contain trailing whitespace, especially inside docstrings where Ruff cannot auto-fix it."
applyTo: "**/*.py"
---
# No Trailing Whitespace on Empty Lines

## Rule

Empty lines — including blank lines inside docstrings — must contain **no characters at all** (not even spaces or tabs).

Ruff's formatter (`ruff format`) removes trailing whitespace from most code, but it does **not** fix blank lines inside docstrings. Violations there must be caught manually or via an editor setting.

## In Docstrings (Priority)

This is where the rule is most easily violated and hardest to auto-fix.

```python
# BAD — blank line between paragraphs has trailing spaces (invisible but present)
def compute(x):
    """Compute the result.
    ·····                   ← trailing spaces on this blank line
    Returns the value.
    """

# GOOD — blank line is truly empty
def compute(x):
    """Compute the result.

    Returns the value.
    """
```

The same applies to blank lines at the top, middle, or bottom of any docstring block.

## In Regular Code

Ruff handles these automatically on format, but do not introduce them in new code:

```python
# BAD
def foo():
····                        ← trailing spaces on blank line between statements
    return 1

# GOOD
def foo():

    return 1
```

## Editor Configuration

Enable "trim trailing whitespace on save" in your editor to prevent introducing violations:

- **VS Code**: `"files.trimTrailingWhitespace": true` in `settings.json`

> **Note:** This setting does not affect whitespace inside string/docstring literals in most editors. Always visually verify blank lines inside multiline strings.
