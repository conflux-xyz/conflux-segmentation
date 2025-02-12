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

## Lint, Format, Type Check, Test

Then, to run linting, fix formatting, and perform type checking:

```bash
poetry run ruff check .
poetry run ruff format .
poetry run mypy .
```

And to run tests:

```shell
poetry run pytest tests
```

## Publishing

To publish version x.y.z, perform the following:

1. Change the `version` in [`pyproject.toml`](./pyproject.toml) to `"x.y.z"`.
2. Once that version is on the `main` branch (e.g. via PR), then tag the branch with `vx.y.z`:
    ```shell
    git tag -a vx.y.z HEAD
    git push origin vx.y.z
    ```
3. Publish a release on GitHub. This will trigger a workflow to publish to PyPI.