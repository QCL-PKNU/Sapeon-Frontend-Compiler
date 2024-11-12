#!/usr/bin/python3

import os
import subprocess
import argparse


def get_root_dir():
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    dirs = scripts_dir.split("/")[:-1]
    return "/".join(dirs)


def build_simulator():
    root_dir = get_root_dir()
    build_script_path = root_dir + "/scripts/build_simulator.sh"
    try:
        subprocess.run([build_script_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Command returned non-zero exit status:", e.returncode)


def setup_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-path",
        dest="model_path",
        required=True,
        type=os.path.abspath,
        help="spear graph model binary path",
    )
    parser.add_argument(
        "--preprocess-cfg-path",
        dest="preprocess_cfg_path",
        required=True,
        type=os.path.abspath,
        help="preprocess config path",
    )
    parser.add_argument(
        "--calibration-method",
        dest="calibration_method",
        required=True,
        type=str,
        help="calibration mode for calibration [max|percentile|entropy|entropy2]",
    )
    parser.add_argument(
        "--calibration-percentile",
        dest="calibration_percentile",
        type=float,
        help="percentile value for percentile calibration. range is 0 to 1",
    )
    parser.add_argument(
        "--calibration-image-dir",
        dest="calibration_image_dir",
        required=True,
        type=os.path.abspath,
        help="image directory path for calibration",
    )
    parser.add_argument(
        "--calibration-batch-size",
        dest="calibration_batch_size",
        required=True,
        type=int,
        help="batch number for calibration. histogram range will be setted by first batch.",
    )
    parser.add_argument(
        "--validation-image-dir",
        dest="validation_image_dir",
        required=True,
        type=os.path.abspath,
        help="image directory path for validation. image directory have to divided by it's classes",
    )


def run_simulator(
    model_path: str,
    preprocess_cfg_path: str,
    calib_method: str,
    calib_image_dir: str,
    calib_batch_size: int,
    valid_image_dir: str,
    calib_percentile: float = None,
):
    root_dir = get_root_dir()
    simulator_path = root_dir + "/simulator"

    simulator_args = [
        "--backend",
        "cpu",
        "--model-path",
        model_path,
        "--graph-type",
        "spear_graph",
        "--dump-level",
        "none",
        "--preprocess-config-path",
        preprocess_cfg_path,
        "--calib",
        "--calibration-method",
        calib_method,
        "--calibration-image-dir",
        calib_image_dir,
        "--calibration-batch-size",
        str(calib_batch_size),
        "--quant",
        "--quant-simulator",
        "x220",
        "--valid",
        "--validation-image-dir",
        valid_image_dir,
    ]

    if (calib_percentile is not None) and (calib_method == "percentile"):
        simulator_args.append("--calibration-percentile")
        simulator_args.append(str(calib_percentile))

    print("command : ./simulator " + " ".join(simulator_args))

    try:
        return subprocess.run(
            [simulator_path] + simulator_args,
            check=True,
            # capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print("Command returned non-zero exit status:", e.returncode)


def handle_stdout_simulator(stdout: str) -> str:
    valid_lines = [
        line
        for line in stdout.splitlines()
        if (("validation_delegate" in line) and ("%" in line))
    ]
    accuracy = valid_lines[-1].split(",")[-1]
    return accuracy


if __name__ == "__main__":
    build_simulator()

    parser = argparse.ArgumentParser()
    setup_arguments(parser)

    args = parser.parse_args()

    uses_calib_pc = (args.calibration_percentile is not None) and (
        args.calibration_method == "percentile"
    )
    calib_pc = args.calibration_percentile if uses_calib_pc else None

    result = run_simulator(
        model_path=args.model_path,
        preprocess_cfg_path=args.preprocess_cfg_path,
        calib_method=args.calibration_method,
        calib_image_dir=args.calibration_image_dir,
        calib_batch_size=args.calibration_batch_size,
        valid_image_dir=args.validation_image_dir,
        calib_percentile=calib_pc,
    )

    valid_accuracy = handle_stdout_simulator(result.stdout.decode())
    pc_str = f"(pc : {calib_pc})" if calib_pc is not None else ""

    print(
        f"Finished validate using {args.calibration_method} method. {pc_str}\nvalidation accuracy : {valid_accuracy}\n\n"
    )
