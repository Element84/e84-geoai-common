# geoai-common
Contains Geospatial AI/ML related code

## Developing

1. Checkout the code.
1. Create/activate your Python environment of choice.
1. Install uv: `pip install uv`.
1. Install dependencies: `uv pip install -r pyproject.toml`.
1. Install dev dependencies: `uv pip install -r pyproject.toml --extra dev`.
1. Run `pre-commit install` to install pre-commit hooks.
1. Configure your editor for realtime linting:
	- For VS Code:
		- Set the correct Python environment for the workspace via `ctrl+shift+P` > `Python: Select Interpreter`.
		- Install the Pylance and Ruff extensions.
1. Make changes.
1. Verify linting passes `scripts/lint.sh`.
1. Verify tests pass `scripts/test.sh`.
1. Commit and push your changes.
	- Note: if using Gitkraken, launch it from the terminal (with `gitkraken`) with the correct python environment activated to ensure that it can use the pre-commit hooks.
