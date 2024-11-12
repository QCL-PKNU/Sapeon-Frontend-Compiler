import torch
import os
from torchmetrics.functional import peak_signal_noise_ratio


def get_darknet_results(path, model_name):
    darknet_results = []
    with open(path, "r") as darknet:
        line = darknet.readline()
        if line.count("input") > 0:
            value = line.strip().split("input")[1].strip().split(" ")[0]
            darknet_results.append(float(value))
        while (True):
            output_layer_condition = True
            line = darknet.readline()
            if not line:
                break
            if model_name == "resnet50":
                output_layer_condition = line.count("ROUTE") == 0 and line.count("SOFTMAX") == 0
            elif model_name == "resnet50_no_connected":
                output_layer_condition = line.count("ROUTE") == 0 and line.count("SOFTMAX") == 0
            elif model_name == "yolov2":
                not_selected_layers = [0, 2, 6, 10, 25, 26, 31]
                if int(line.split()[0]) in not_selected_layers:
                    output_layer_condition = False
            elif model_name == "yolov3":
                output_layer_condition1 = line.count("YOLO") == 0
                output_layer_condition2 = line.count(" 83 ") == 0 or line.count("ROUTE") == 0
                output_layer_condition3 = line.count(" 95 ") == 0 or line.count("ROUTE") == 0
                output_layer_condition = output_layer_condition1 and output_layer_condition2 and output_layer_condition3
            elif model_name == "keras":
                not_selected_layers = [1, 4, 8, 12, 16, 21]
                if int(line.split()[0]) in not_selected_layers:
                    output_layer_condition = False

            if output_layer_condition:
                value = line.strip().split("output")[1].strip().split(" ")[0]
                darknet_results.append(float(value))
    if model_name in ["mobilenet"]:
        darknet_results = darknet_results[:len(darknet_results)-1]
    return darknet_results


def get_aix_results(path, model_name):
    aix_results = []
    with open(path, "r") as aix:
        num_line = 1
        while (True):
            output_layer_condition = True
            line = aix.readline()
            if not line:
                break
            if model_name == "resnet50":
                not_selected_lines = [8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 60, 65, 70, 76, 81, 86]
                if num_line in not_selected_lines:
                    output_layer_condition = False
            if model_name == "mobilenet":
                not_selected_lines = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
                if num_line in not_selected_lines:
                    output_layer_condition = False
            if output_layer_condition:
                value = line.strip().split()[3].strip()
                aix_results.append(float(value))
            num_line += 1
        return aix_results


def switch_index(aix_results: list):
    val = aix_results[3]
    del aix_results[3]
    aix_results.insert(6, val)

    val = aix_results[16]
    del aix_results[16]
    aix_results.insert(19, val)

    val = aix_results[33]
    del aix_results[33]
    aix_results.insert(36, val)

    return aix_results


if __name__ == "__main__":
    project_paths = os.path.abspath(os.path.dirname(__file__))
    print(project_paths)
    project_paths = project_paths.split("/")
    project_path = "/".join(project_paths[:len(project_paths)-1])
    model_name = "resnet50_no_connected"
    table_path = "/" + model_name + "/" + model_name + "_calib_tbl_entropy_500"
    aix_calib_table_path = project_path + '/calibration-table-test-spear-cudnn' + table_path
    darknet_calib_table_path = project_path + '/calibration-table-darknet-spear' + table_path
    model_prefix = model_name.split("_")[0]
    darknet_results = get_darknet_results(darknet_calib_table_path, model_name)
    aix_results = get_aix_results(aix_calib_table_path, model_name)
    aix_results = switch_index(aix_results)
    aix_tensor = torch.tensor(aix_results)
    darknet_tensor = torch.tensor(darknet_results)
    print("aix")
    print(aix_results)
    print("darknet")
    print(darknet_results)
    psnr = peak_signal_noise_ratio(aix_tensor, darknet_tensor).item()
    print('{0:0.2f} dB'.format(psnr))
