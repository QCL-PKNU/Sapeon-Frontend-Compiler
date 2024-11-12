#!/bin/bash

set -ex

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
./scripts/compile_proto.sh

rm -rf build
mkdir -p build
cd build
# cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc)
popd
