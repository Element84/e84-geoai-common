#!/bin/bash

####################################################################################################
# Runs the tests.
#
# Testing against LLMs will use canned responses unless --use-real-bedrock-client is passed in which
# case the tests will be run against the real bedrock. It assumes AWS credentials have been
# configured in that case.
#
# Any extra args to this script will be passed on to the pytest command. This can be used to
# run specific tests e.g.:
# scripts/test.sh tests/llm/models/test_claude.py -k test_tool_use
#
####################################################################################################

set -e -o pipefail

USE_REAL_BEDROCK_CLIENT=false
EXTRA_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--use-real-bedrock-client" ]]; then
    USE_REAL_BEDROCK_CLIENT=true
  else
    EXTRA_ARGS+=("$arg")
  fi
done

export USE_REAL_BEDROCK_CLIENT

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -vv -rA --log-cli-level=INFO "${EXTRA_ARGS[@]}"
