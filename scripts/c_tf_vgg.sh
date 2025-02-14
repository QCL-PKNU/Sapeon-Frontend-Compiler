#!/bin/bash

# Compile the VGG model
cd .. && make frontend MODEL_NAME=vgg16.pb \
                     MD_FILE=tensorflow_sample.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text