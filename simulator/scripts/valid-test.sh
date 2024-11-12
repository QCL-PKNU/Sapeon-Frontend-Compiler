#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm dump/*
rm results/*

BACKEND=cpu
MODEL_PATH=examples/n05.sp

./scripts/build_simulator.sh
./simulator --backend $BACKEND \
            --model-path $MODEL_PATH \
            --graph-type spear_graph \
            --dump-level default \
            --preprocess-config-path configs/preprocess/resnet50_torch.json \
            --valid \
            --validation-image-dir images/imagenet_validation
popd
