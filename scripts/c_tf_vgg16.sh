#!/bin/bash

# Compile the Vgg model
cd .. && make MODEL_NAME=vgg16.pb \
                     MD_FILE=tensorflow.md \
                     CALIB_FILE=mobilenet_calib.tbl \
                     GRAPH_FORMAT=binary