#!/bin/bash

# Compile the VGG model
cd .. && make frontend MODEL_NAME=vgg_model.pth \
                     MD_FILE=torch/resnet.md \
                     CALIB_FILE=resnet50_v1_imagenet_calib.tbl \
                     GRAPH_FORMAT=text \
                     INPUT_SHAPE="[1,3,224,224]"