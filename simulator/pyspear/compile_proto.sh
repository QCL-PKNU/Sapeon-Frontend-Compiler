#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"
PROTO_FILE_DIR="proto_files"
PROTO_FILE="spear.proto.e8e8"

pushd $PROJECT_DIR
mkdir -p pyspear/pb2
protoc -I=$PROTO_FILE_DIR --python_out=pyspear/pb2 $PROTO_FILE_DIR/$PROTO_FILE
popd
