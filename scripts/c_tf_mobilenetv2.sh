#!/bin/bash

# Compile the Mobilenet V1 model
cd .. && make MODEL_NAME=mobilenetv2.pb \
                     MD_FILE=tensorflow.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=binary