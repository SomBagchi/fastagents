# FastAgents

FastAgents research prototype.

## Installation

```bash
pip install -e ".[dev]"
```

# Build the Bash sandbox image
```bash
docker build -t fastagents-bash:0.1 docker/
```

## Development

This project uses:
- Black for code formatting
- Ruff for linting
- MyPy for type checking
- Pytest for testing
- Pre-commit for git hooks
