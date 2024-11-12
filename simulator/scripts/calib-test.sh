#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm dump/*
rm results/*

BACKEND=cpu
MODEL_PATH=examples/entropy_test.sp
CALIB_METHOD=entropy

./scripts/build_simulator.sh
./simulator --backend $BACKEND \
            --model-path $MODEL_PATH \
            --graph-type spear_graph \
            --dump-level default \
            --preprocess-config-path configs/preprocess/resnet50.json \
            --calib \
            --calibration-method $CALIB_METHOD \
            --calibration-batch-size 500 \
            --calibration-image-dir images/imagenet \
            --dump-calibration-table \
            --calibration-table-dump-path results/calib-table.txt
popd
