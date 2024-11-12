#!/bin/bash

# PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"
# MODEL_NAME="yolov4"
# MODEL_PATH="models/calib/simulator/${MODEL_NAME}.pb"
# PB_PATH="spear.proto.e8e8"
# PBTXT_PATH="./${MODEL_NAME}.pb.txt"

# pushd $PROJECT_DIR
# protoc --encode \
#   sapeon.SPGraph \
#   --proto_path=proto_files \
#   ${PB_PATH} < ${PBTXT_PATH} >| ${MODEL_PATH}
# popd


# PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"
# MODEL_NAME="aix_graph"
# MODEL_PATH="assets/${MODEL_NAME}.pb"
# PB_PATH="aixh.proto"
# PBTXT_PATH="aix_graph.pb.txt"

# pushd $PROJECT_DIR
# protoc --encode \
#   aixh.AIXGraph \
#   --proto_path=proto_files \
#   ${PB_PATH} < ${PBTXT_PATH} >| ${MODEL_PATH}
# popd


PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"
MODEL_NAME="aix_graph"
MODEL_PATH="assets/${MODEL_NAME}.pb"
PB_PATH="aixh.proto"
PBTXT_PATH="assets/aix_graph.pbtxt"

pushd $PROJECT_DIR
protoc --encode \
  aixh.AIXGraph \
  --proto_path=proto_files \
  ${PB_PATH} < ${PBTXT_PATH} > ${MODEL_PATH}
popd
