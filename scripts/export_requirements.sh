#!/bin/bash

##################################################
# Exports requirements to a requirements.txt file.
##################################################

set -e -o pipefail

uv export --no-hashes --all-extras --format requirements-txt >requirements.txt
