#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"
MODEL_NAME="yolov4"
MODEL_PATH="models/calib/simulator/${MODEL_NAME}.pb"
PB_PATH="spear.proto.e8e8"
PBTXT_PATH="./${MODEL_NAME}.pb.txt"

pushd $PROJECT_DIR
protoc --decode \
  sapeon.SPGraph \
  --proto_path=proto_files \
  ${PB_PATH} < ${MODEL_PATH} >| ${PBTXT_PATH}
popd
