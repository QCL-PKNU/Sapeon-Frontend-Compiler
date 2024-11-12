#!/usr/bin/python3

import torch
from torchmetrics.functional import peak_signal_noise_ratio
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--first", dest="first_tensor", type=str, help="First tensor file path"
    )
    parser.add_argument(
        "-s", "--second", dest="second_tensor", type=str, help="Second tensor file path"
    )
    args = parser.parse_args()
    first_output = np.fromfile(args.first_tensor, dtype="float32")
    first_tensor = torch.tensor(first_output)

    second_output = np.fromfile(args.second_tensor, dtype="float32")
    second_tensor = torch.tensor(second_output)

    psnr = peak_signal_noise_ratio(first_tensor, second_tensor).item()
    print(f"psnr = {psnr:.2f} dB")


if __name__ == "__main__":
    main()
