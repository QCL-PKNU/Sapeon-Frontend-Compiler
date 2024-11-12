#!/bin/bash

set -ex

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm -rf test_cpu
rm -rf build
mkdir -p build
cd build
cmake -DBUILD_TESTS=ON -DGPU=0 ..
make -j $(nproc)
cd ..
./test_cpu --colorlogtostderr=true --logtostderr=true
popd
