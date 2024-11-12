#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm dump/*
rm results/*

MODEL_PATH=models/fp32/SPgraph/n15.sp

# ./scripts/build_simulator.sh
./simulator --backend cpu \
            --model-path $MODEL_PATH \
            --graph-type spear_graph \
            --dump-level default \
            --preprocess-config-path configs/preprocess/resnet50.json \
            --infer \
            --image-path images/dog_224_nopre.png

popd
