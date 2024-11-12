#!/bin/python3

import torch
import os
from torchmetrics.functional import peak_signal_noise_ratio
import onnxruntime as ort
import numpy as np
import PIL

def run_onnx(model_path):
    ort_sess = ort.InferenceSession(model_path)
    image = PIL.Image.open('images/dog_224_nopre.png')
    image_data = np.array(image).transpose(2, 0, 1)
    input_data = preprocess(image_data)
    input_name = ort_sess.get_inputs()[0].name
    outputs = ort_sess.run(None, {input_name: input_data})
    return outputs[0].flatten()

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def main():
    onnx_output = run_onnx('models/fp32/onnx/n08.onnx')
    sp_output = np.fromfile('dump/network.output', dtype='float32')
    onnx_tensor = torch.tensor(onnx_output)
    sp_tensor = torch.tensor(sp_output)
    psnr = peak_signal_noise_ratio(onnx_tensor, sp_tensor).item()
    print(f'psnr = {psnr:.2f} dB')

if __name__ == "__main__":
    main()