#!/usr/bin/python3

import os
import filecmp

# darknet_quant : simulator
layer_map = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    # 5: Route
    6: 2,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 16,
    17: 17,
    18: 18,
    # 19: Route
    20: 15,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    26: 24,
    27: 25,
    28: 26,
    29: 27,
    30: 28,
    31: 29,
    32: 30,
    33: 31,
    34: 33,
    35: 34,
    36: 35,
    # 37: Route
    38: 32,
    39: 36,
    40: 37,
    41: 38,
    42: 39,
    43: 40,
    44: 41,
    45: 42,
    46: 43,
    47: 44,
    48: 45,
    49: 46,
    50: 47,
    51: 48,
    52: 49,
    53: 50,
    54: 51,
    55: 52,
    56: 53,
    57: 54,
    58: 55,
    59: 56,
    60: 58,
    61: 59,
    62: 60,
    # 63: Route
    64: 57,
    65: 61,
    66: 62,
    67: 63,
    68: 64,
    69: 65,
    70: 66,
    71: 67,
    72: 68,
    73: 69,
    74: 70,
    75: 71,
}


def compare_file(f1, f2):
    f1_name = os.path.basename(f1)
    f2_name = os.path.basename(f2)
    if not os.path.exists(f1):
        f1_name += "?"
    if not os.path.exists(f2):
        f2_name += "?"

    if not os.path.exists(f1) or not os.path.exists(f2):
        result = "No file"
        ret = 1
    elif not filecmp.cmp(f1, f2, shallow=False):
        result = "Diff"
        ret = 2
    else:
        result = "OK"
        ret = 0

    print(f"{result:>8s}:   {f1_name:25s} |   {f2_name:25s}")
    return ret


def main():
    cwd = os.path.abspath(os.path.dirname(__file__))
    darknet_dump_path = cwd + "/../dump_darknet"
    simulator_dump_path = cwd + "/../dump"

    ok = 0
    no_file = 0
    diff = 0

    no_file_list = []
    diff_list = []

    print(" Compare:   darknet_quant             |   simulator")
    print("--------------------------------------+----------------------------")
    for darknet_dump in sorted(os.listdir(darknet_dump_path)):
        simulator_dump = darknet_dump
        for key, value in layer_map.items():
            key_str = f"{key:03d}"
            value_str = f"{value:03d}"
            if key_str in simulator_dump:
                simulator_dump = simulator_dump.replace(key_str, value_str)
                break

        result = compare_file(
            os.path.join(darknet_dump_path, darknet_dump),
            os.path.join(simulator_dump_path, simulator_dump),
        )
        if result == 0:
            ok += 1
        elif result == 1:
            no_file += 1
            no_file_list.append(darknet_dump)
        else:
            diff += 1
            diff_list.append(darknet_dump)

    print()
    print(f"Total: {ok + no_file + diff}")
    print(f"OK: {ok}, No file: {no_file}, Diff: {diff}")


if __name__ == "__main__":
    main()
