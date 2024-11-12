#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm dump/*
rm results/*

BACKEND=cpu
# MODEL_PATH=models/quant/simulator/resnet50_no_connected_for_quant.pb
MODEL_PATH=examples/resnet50_quant.pb

./scripts/build_simulator.sh
./simulator --backend $BACKEND \
            --model-path $MODEL_PATH \
            --graph-type spear_graph \
            --dump-level debug \
            --preprocess-config-path configs/preprocess/resnet50_no_connected.json \
            --quant \
            --quant-simulator x220 \
            --infer \
            --image-path images/dog_224_nopre.png

./scripts/compare-quant-darknet.py > results/quant-compare-result.txt
popd
