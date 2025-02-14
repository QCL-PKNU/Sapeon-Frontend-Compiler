#!/bin/bash

# Compile the ResNet50 model
cd .. && make MODEL_NAME=resnet50-v1-7.onnx \
                     MD_FILE=onnx.md \
                     CALIB_FILE=resnet50_v1_imagenet_calib.tbl \
                     GRAPH_FORMAT=text