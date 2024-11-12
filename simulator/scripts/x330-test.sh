#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
rm dump/*
rm results/*
mkdir -p results_x330
rm -rf dump_simulator_x330
mkdir -p dump_simulator_x330

quant_cfgs=(
  "all_nf8.cfg"
  "resnet50.base_nf8_nf8u.cfg"
  "resnet50.base_nf8_nf8u.in_nf10.out_nf16.cfg"
  "resnet50.base_nf8_nf8u.in_nf16.cfg"
  "resnet50.base_nf8_nf8u.io_nf16.cfg"
  "resnet50.base_nf8_only.cfg"
  "resnet50.base_nf8_only.in_nf16.cfg"
  "resnet50.base_nf8_only.io_nf16.cfg"
  "resnet50.dtype.01.cfg"
  "resnet50.dtype.02.cfg"
  "resnet50.dtype.03.cfg"
  "resnet50.dtype.04.cfg"
  "resnet50.dtype.05.cfg"
  "resnet50.dtype.06.cfg"
  "resnet50.dtype.07.cfg"
  "resnet50.dtype.08.cfg"
  "resnet50.dtype.09.cfg"
  "resnet50.dtype.10.cfg"
  "resnet50.dtype.11.cfg"
  "resnet50.dtype.12.cfg"
  "resnet50.ebias.1.cfg"
  "resnet50.ebias.2.cfg"
  "resnet50.ebias.3.cfg"
  "resnet50.ebias.4.cfg"
  "resnet50.fcalib.1.cfg"
  "resnet50.fcalib.2.cfg"
  "resnet50.fcalib.3.cfg"
  "resnet50.fcalib.4.cfg"
  "resnet50.fcalib.5.cfg"
  "resnet50.fcalib.6.cfg"
  "resnet50.rmode.1.cfg"
  "resnet50.rmode.2.cfg"
  "resnet50.rmode.3.cfg"
  "resnet50.wcalib.1.cfg"
  "resnet50.wcalib.2.cfg"
  "resnet50.wcalib.3.cfg"
)

BACKEND=cpu
# MODEL_PATH=models/quant/simulator/resnet50_no_connected_for_quant.pb
MODEL_PATH=examples/resnet50_quant.pb

./scripts/build_simulator.sh

for cfg in "${quant_cfgs[@]}"
do
  rm -rf dump
  rm -rf results
  mkdir -p dump
  mkdir -p results
  ./simulator --backend $BACKEND \
              --model-path $MODEL_PATH \
              --graph-type spear_graph \
              --dump-level debug \
              --preprocess-config-path configs/preprocess/resnet50_no_connected.json \
              --quant \
              --quant-simulator x330 \
              --quant-cfg-path configs/quantization/x330/"${cfg}" \
              --infer \
              --image-path images/dog_224_nopre.png

  ./scripts/compare-quant-x330.py --cfg "${cfg}" > results_x330/"${cfg}"-result.txt
  mv dump dump_simulator_x330/"${cfg}"
done
popd
