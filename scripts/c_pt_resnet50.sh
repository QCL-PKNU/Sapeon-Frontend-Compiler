#!/bin/bash

# Compile the ResNet50 model
cd .. && make MODEL_NAME=resnet50.pth \
                     MD_FILE=torch/resnet.md \
                     CALIB_FILE=resnet50_v1_imagenet_calib.tbl \
                     GRAPH_FORMAT=text \
                     INPUT_SHAPE="[1,3,224,224]"
