#!/bin/bash

# Compile the VGG model
cd .. && make frontend MODEL_NAME=vgg16-7.onnx \
                     MD_FILE=onnx_sample.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text