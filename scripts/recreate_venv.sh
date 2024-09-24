#!/bin/bash

####################################################################################################
# Recreates the virtual environment with frozen dependencies.
####################################################################################################

set -e -o pipefail

rm -rf .venv

uv venv

uv pip install -r requirements.txt
