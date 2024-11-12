#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm -rf tests/test_data
mkdir -p tests/test_data

touch tests/test_data/existing_empty_file

# onnx and onnxruntime version for python3.6
pip install onnx==1.10 onnxruntime==1.10
python3 scripts/cpu_operation_test.py

popd
