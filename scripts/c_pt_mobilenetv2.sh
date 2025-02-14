#!/bin/bash

# Compile the MobileNetV2 model
cd .. && make MODEL_NAME=mobilenet_v2.pth \
                     MD_FILE=torch/mobilenet.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text \
                     INPUT_SHAPE="[1,3,224,224]"