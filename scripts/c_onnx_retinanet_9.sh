#!/bin/bash

# Compile the RetinaNet-9 model
cd .. && make frontend MODEL_NAME=retinanet-9.onnx \
                     MD_FILE=onnx_sample.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text