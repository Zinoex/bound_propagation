# Development Guide

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management and builds.

## Installation

First, install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS/Linux with Homebrew:
```bash
brew install uv
```

## Setup

The project will automatically use the Python version specified in `.python-version` (Python 3.11).

### Install the package in development mode

```bash
# Install package with runtime dependencies
uv pip install -e .

# Install with development dependencies (includes sphinx, testing tools, etc.)
uv pip install -e ".[dev]"
```

### Create a virtual environment (optional)

```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

## Building

Build the package:
```bash
uv build
```

This will create both wheel and source distributions in the `dist/` directory.

## Dependency Management

All dependencies are managed in `pyproject.toml`:
- Runtime dependencies: `[project.dependencies]`
- Development dependencies: `[project.optional-dependencies.dev]`

To add a new dependency:
1. Edit `pyproject.toml` and add it to the appropriate section
2. Run `uv pip install -e ".[dev]"` to install the new dependency

## Migration Notes

This project has been migrated to use modern pyproject.toml configuration with uv:
- All configuration is now in `pyproject.toml` following PEP 621 standards
- Uses uv's native build backend for fast, efficient builds
