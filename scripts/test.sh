#!/bin/bash

set -e -o pipefail

PYTHONPATH=src:tests pytest
