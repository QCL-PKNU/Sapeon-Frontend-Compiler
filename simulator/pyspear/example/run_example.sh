#!/bin/bash

EXAMPLE_DIR="$( cd "$( dirname "$0" )" && pwd -P )"

MODEL_NAME="pyspear-resnet18_b-1"
MODEL_PATH="./${MODEL_NAME}.sp"
PB_PATH="../../proto_files/spear.proto.e8e8"
PBTXT_PATH="./${MODEL_NAME}.pb.txt"

pushd $EXAMPLE_DIR
python3 resnet18_b-1.py

protoc --decode \
  sapeon.simulator.SPGraph \
  --proto_path=../../proto_files \
  ${PB_PATH} < ${MODEL_PATH} >| ${PBTXT_PATH}
popd
