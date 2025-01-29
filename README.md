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

## Contributions

We are happy to take contributions! It is best to get in touch with the maintainers about larger features or design changes *before* starting the work, as it will make the process of accepting changes smoother.

## Contributor License Agreement (CLA)

Everyone who contributes code to E84 Geoai Common will be asked to sign a CLA, which is based off of the Apache CLA.

- Download a copy of **one of** the following from the `docs/cla` directory in this repository:

  - Individual Contributor (You're using your time): `2025_1_29-E84-Geoai-Common-Open-Source-Contributor-Agreement-Individual.pdf`
  - Corporate Contributor (You're using company time): `2025_1_29-E84-Geoai-Common-Open-Source-Contributor-Agreement-Corporate.pdf`

- Sign the CLA -- either physically on a printout or digitally using appropriate PDF software.

- Send the signed CLAs to Element 84 via **one of** the following methods:

  - Emailing the document to contracts@element84.com
  - Mailing a hardcopy to: ``Element 84, 210 N. Lee Street Suite 203 Alexandria, VA 22314, USA``.
