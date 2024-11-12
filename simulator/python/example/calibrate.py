import os
import sys

import numpy as np
import cv2

project_root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../.."
build_path = os.path.join(project_root_dir, "build")
print(build_path)
sys.path.insert(0, build_path)

import sapeon.simulator as spsim

model_path = os.path.join(project_root_dir, "mobilenet_v1_b-1.onnx")
img_dir = os.path.join(project_root_dir, "images/images5")
img_paths = []
for img_name in ["dog.jpg", "eagle.jpg", "giraffe.jpg", "kite.jpg", "person.jpg"]:
    img_paths.append(os.path.join(img_dir, img_name))


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_npy = np.asarray(img, dtype=np.float32)
    img_npy = img_npy.transpose([2, 0, 1])
    img_npy /= 255.0
    return img_npy


input_tensor = None
for img_path in img_paths:
    img = cv2.imread(img_path)
    img_npy = preprocess(img)
    if input_tensor is None:
        input_tensor = np.array([img_npy])
    else:
        input_tensor = np.vstack((input_tensor, [img_npy]))

calibrator = spsim.make_calibrator(
    model_path, spsim.CalibrationMethod.Percentile, 0.999
)
calibrator.collect(input_tensor)
calibrator.compute_range().as_textfile("calib-table.txt")
