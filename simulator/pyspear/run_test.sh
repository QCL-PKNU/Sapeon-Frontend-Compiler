#!/bin/bash

PYSPEAR_DIR="$( cd "$( dirname "$0" )" && pwd -P )"

pushd $PYSPEAR_DIR/test

pytest

popd
