#!/bin/bash

# Compile the MobileNet V2 model
cd .. && make frontend MODEL_NAME=mobilenetv2-7.onnx \
                     MD_FILE=onnx_sample.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text