#!/bin/bash

PYSPEAR_DIR="$( cd "$( dirname "$0" )" && pwd -P )"

MODEL_NAME="pyspear"
MODEL_PATH="./${MODEL_NAME}.pb"
PB_PATH="../proto_files/spear.proto.e8e8"
PBTXT_PATH="./${MODEL_NAME}.pb.txt"

pushd $PYSPEAR_DIR
python3 main.py

protoc --decode \
  sapeon.simulator.SPGraph \
  --proto_path=../proto_files \
  ${PB_PATH} < ${MODEL_PATH} >| ${PBTXT_PATH}
popd
