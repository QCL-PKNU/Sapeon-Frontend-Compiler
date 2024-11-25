#!/bin/bash

# Compile the ResNet50 model
cd .. && make frontend MODEL_NAME=resnet50.pb \
                     MD_FILE=tensorflow_sample.md \
                     CALIB_FILE=resnet50_v1_imagenet_calib.tbl \
                     GRAPH_FORMAT=text