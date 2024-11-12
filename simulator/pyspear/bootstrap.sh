#!/bin/bash

PYSPEAR_DIR="$( cd "$( dirname "$0" )" && pwd -P )"

pushd $PYSPEAR_DIR
pip3 install -e .
./compile_proto.sh
popd
