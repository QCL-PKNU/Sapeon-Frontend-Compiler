#!/bin/bash

# Compile the Mobilenet V1 model
cd .. && make frontend MODEL_NAME=mobilenetv2.pb \
                     MD_FILE=tensorflow_sample.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=text