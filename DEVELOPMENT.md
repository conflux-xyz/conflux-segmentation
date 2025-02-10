# Development

## Set-up

This repo uses [Poetry](https://python-poetry.org/) for dependency management.

After cloning the repo, perform the following:

Create a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

Install all dependencies:

```bash
poetry install --all-extras
```

Then, to run linting, fix formatting, and perform type checking:

```bash
ruff check .
ruff format .
mypy .
```