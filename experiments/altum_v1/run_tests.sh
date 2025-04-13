#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the experiments directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(dirname $DIR)

# Run the tests with verbose output
pytest -v $DIR/tests 