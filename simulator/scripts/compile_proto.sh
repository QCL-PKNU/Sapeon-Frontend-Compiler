#!/bin/bash

set -e

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

# graph_index
proto_files=("aixh.proto" "spear.proto.v1.2" "spear.proto.e8e8")
cc_files=("aixh.pb.cc" "spear.proto.v1.2.pb.cc" "spear.proto.e8e8.pb.cc")
h_files=("aixh.pb.h" "spear.proto.v1.2.pb.h" "spear.proto.e8e8.pb.h")

pushd ${PROJECT_DIR}/proto_files
for graph_index in 0 2
do
    protoc ${proto_files[graph_index]} --cpp_out=.
    mv ${cc_files[graph_index]} ../src/core/
    mv ${h_files[graph_index]} ../include/
done
popd
