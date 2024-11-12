#!/usr/bin/python3

import os
import subprocess
import torch
from torchmetrics.functional import peak_signal_noise_ratio
import numpy as np
import struct
import shutil
import re

RESNET50_QUANT_CFGS = [
    "all_nf8.cfg",
    "resnet50.base_nf8_nf8u.cfg",
    "resnet50.base_nf8_nf8u.in_nf10.out_nf16.cfg",
    "resnet50.base_nf8_nf8u.in_nf16.cfg",
    "resnet50.base_nf8_nf8u.io_nf16.cfg",
    "resnet50.base_nf8_only.cfg",
    "resnet50.base_nf8_only.in_nf16.cfg",
    "resnet50.base_nf8_only.io_nf16.cfg",
    "resnet50.dtype.01.cfg",
    "resnet50.dtype.02.cfg",
    "resnet50.dtype.03.cfg",
    "resnet50.dtype.04.cfg",
    "resnet50.dtype.05.cfg",
    "resnet50.dtype.06.cfg",
    "resnet50.dtype.07.cfg",
    "resnet50.dtype.08.cfg",
    "resnet50.dtype.09.cfg",
    "resnet50.dtype.10.cfg",
    "resnet50.dtype.11.cfg",
    "resnet50.dtype.12.cfg",
    "resnet50.ebias.1.cfg",
    "resnet50.ebias.2.cfg",
    "resnet50.ebias.3.cfg",
    "resnet50.ebias.4.cfg",
    "resnet50.fcalib.1.cfg",
    "resnet50.fcalib.2.cfg",
    "resnet50.fcalib.3.cfg",
    "resnet50.fcalib.4.cfg",
    "resnet50.fcalib.5.cfg",
    "resnet50.fcalib.6.cfg",
    "resnet50.rmode.1.cfg",
    "resnet50.rmode.2.cfg",
    "resnet50.rmode.3.cfg",
    "resnet50.wcalib.1.cfg",
    "resnet50.wcalib.2.cfg",
    "resnet50.wcalib.3.cfg",
    "test.bf16.cfg",
    "test.fp32.cfg",
    "test.nf8.cfg",
    "test.nf8.ebias0.cfg",
    "test.nf8.fcalib.add.cfg",
    "test.nf8.fcalib.min.cfg",
    "test.nf8.fcalib.set.cfg",
    "test.nf16.cfg",
]

TEST_QUANT_CFGS = [
    # "test.bf16.cfg",
    # "test.nf8.cfg",
    # "test.nf8.ebias0.cfg",
    # "test.nf8.fcalib.add.cfg",
    "test.nf8.fcalib.min.cfg",
    "test.nf8.fcalib.set.cfg",
    # "test.nf16.cfg",
]

MODEL_NAMES = [
    "n01",
    "n02",
    "n03",
    "n04",
    "n05",
    "n06",
    "n07",
    "n08",
    "n09",
    "n10",
    "n11",
    "n12",
    "n13",
    "n14",
    "n15",
    "n16",
    "n17",
    "n18",
    "n19",
    "n20",
    # "n21",
    # "n22",
]

MODEL_INPUT_DIMENSIONS = [
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    224,
    608,
    416,
    640,
    640,
    640,
    # 256,
    # 128,
]


def get_root_dir():
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    dirs = scripts_dir.split("/")[:-1]
    return "/".join(dirs)


def build_simulator():
    root_dir = get_root_dir()
    build_script_path = root_dir + "/scripts/build_simulator.sh"
    subprocess.run([build_script_path])


def setup_quant_max_directories(model_name: str):
    root_dir = get_root_dir()

    dump_dir = root_dir + "/dump"
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir)

    quant_max_dir = root_dir + "/quant_max/" + model_name
    if os.path.exists(quant_max_dir):
        shutil.rmtree(quant_max_dir)
    os.makedirs(quant_max_dir)


def setup_dump_directories(model_name: str, quant_cfg: str):
    root_dir = get_root_dir()

    dump_layer_output_dir = (
        root_dir + "/network-outputs-x330/" + quant_cfg + "/" + model_name
    )
    if os.path.exists(dump_layer_output_dir):
        shutil.rmtree(dump_layer_output_dir)
    os.makedirs(dump_layer_output_dir)

    psnr_dump_dir = root_dir + "/psnr-layers/" + quant_cfg + "/" + model_name
    if os.path.exists(psnr_dump_dir):
        shutil.rmtree(psnr_dump_dir)
    os.makedirs(psnr_dump_dir)


def create_tensor(path: str) -> torch.Tensor:
    with open(path, "r") as f:
        content = f.read()
    hex_list = content.split()
    float_list = [
        struct.unpack("!f", bytes.fromhex(hex_str))[0] for hex_str in hex_list
    ]
    output = np.array(float_list, dtype=np.float32)
    return torch.tensor(output)


def measure_psnr(model_name: str, quant_cfg: str):
    root_dir = get_root_dir()
    dump_dir = root_dir + "/network-outputs-x330"
    ref_output_dir = dump_dir + "/test.fp32.cfg/" + model_name
    target_output_dir = dump_dir + "/" + quant_cfg + "/" + model_name
    psnr_dump_path = (
        root_dir + "/psnr-layers/" + quant_cfg + "/" + model_name + "/psnr.layer.txt"
    )

    pattern = r"layer(\d{3})\.output"
    matched_files = [
        file for file in os.listdir(ref_output_dir) if re.match(pattern, file)
    ]
    sorted_files = sorted(
        matched_files, key=lambda x: int(re.search(pattern, x).group(1))
    )

    with open(psnr_dump_path, "w") as dump:
        for output_filename in sorted_files:
            idx_layer = int(re.search(pattern, output_filename).group(1))
            ref_tensor = create_tensor(ref_output_dir + "/" + output_filename)
            target_tensor = create_tensor(target_output_dir + "/" + output_filename)
            psnr = peak_signal_noise_ratio(ref_tensor, target_tensor).item()
            dump.write(f"layer{idx_layer:03d}.output psnr = {psnr:.2f}\n")
            if output_filename is sorted_files[-1]:
                print(f"\n\nNetwork output's psnr is {psnr:.2f} dB")


def run_simulator(
    model_name: str,
    model_input_dim: int,
    preprocess_cfg: str,
    quant_cfg: str,
    uses_quant_max: bool = False,
):
    root_dir = get_root_dir()
    simulator_path = root_dir + "/simulator"
    model_path = root_dir + "/examples/SPgraph/" + model_name + ".sp"
    dump_dir = root_dir + "/network-outputs-x330/" + quant_cfg + "/" + model_name
    preprocess_cfg_path = root_dir + "/configs/preprocess/" + preprocess_cfg
    image_path = root_dir + "/images/dog_png/dog_" + str(model_input_dim) + ".png"
    quant_cfg_path = root_dir + "/configs/quantization/x330/" + quant_cfg

    args = [
        "--backend",
        "cpu",
        "--model-path",
        model_path,
        "--graph-type",
        "spear_graph",
        "--dump-level",
        "default",
        "--dump-dir",
        dump_dir,
        "--preprocess-config-path",
        preprocess_cfg_path,
        "--quant",
        "--quant-simulator",
        "x330",
        "--quant-cfg-path",
        quant_cfg_path,
        "--infer",
        "--image-path",
        image_path,
    ]

    if uses_quant_max:
        quant_max_path = root_dir + "/quant_max/" + model_name + "/quant.max"
        args.append("--quant-max-path")
        args.append(quant_max_path)

    print("./simulator " + " ".join(args))

    subprocess.run([simulator_path] + args)


def collect_simulator(model_name: str, preprocess_cfg: str):
    root_dir = get_root_dir()
    simulator_path = root_dir + "/simulator"
    model_path = root_dir + "/examples/SPgraph/" + model_name + ".sp"
    dump_dir = root_dir + "/dump"
    preprocess_cfg_path = root_dir + "/configs/preprocess/" + preprocess_cfg
    quant_max_path = root_dir + "/quant_max/" + model_name + "/quant.max"
    images_dir = root_dir + "/images/imagenet"

    args = [
        "--backend",
        "cpu",
        "--model-path",
        model_path,
        "--graph-type",
        "spear_graph",
        "--dump-level",
        "default",
        "--dump-dir",
        dump_dir,
        "--preprocess-config-path",
        preprocess_cfg_path,
        "--collect",
        "--collect-quant-max-dump-path",
        quant_max_path,
        "--collect-image-dir",
        images_dir,
    ]

    print("./simulator " + " ".join(args))

    subprocess.run([simulator_path] + args)


if __name__ == "__main__":
    build_simulator()

    preprocess_cfg = "resnet50_torch.json"
    ref_fp32_cfg = "test.fp32.cfg"
    for model_name, input in zip(MODEL_NAMES, MODEL_INPUT_DIMENSIONS):
        # Dump Reference fp32 outputs
        if model_name == "n16":
            preprocess_cfg = "no_preprocess.json"
        setup_dump_directories(model_name, ref_fp32_cfg)
        run_simulator(
            model_name=model_name,
            model_input_dim=input,
            preprocess_cfg=preprocess_cfg,
            quant_cfg=ref_fp32_cfg,
        )
        setup_quant_max_directories(model_name)
        collect_simulator(model_name=model_name, preprocess_cfg=preprocess_cfg)
        for quant_cfg in TEST_QUANT_CFGS:
            # Dump Target outputs
            setup_dump_directories(model_name, quant_cfg)
            run_simulator(
                model_name=model_name,
                model_input_dim=input,
                preprocess_cfg=preprocess_cfg,
                quant_cfg=quant_cfg,
                uses_quant_max=True,
            )
            measure_psnr(model_name, quant_cfg)
            print(
                f"Finished measuring PSNR for each layer's output using {model_name} model with {quant_cfg}.\n\n"
            )
