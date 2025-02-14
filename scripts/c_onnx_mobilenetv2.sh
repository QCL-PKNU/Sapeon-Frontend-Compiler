#!/bin/bash

# Compile the MobileNet V2 model
cd .. && make MODEL_NAME=mobilenetv2-7.onnx \
                     MD_FILE=onnx.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=binary