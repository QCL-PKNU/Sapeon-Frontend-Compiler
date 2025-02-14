#!/bin/bash

# Compile the VGG model
cd .. && make MODEL_NAME=vgg16-7.onnx \
                     MD_FILE=onnx.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text