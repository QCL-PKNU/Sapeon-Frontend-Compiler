# 2022 Spring Undergraduate Research Opportunities Program

## Current Status

### AIXGraph Simulator Specification

- Supported model input
   - AIXGraph
- Supported backends
   - NVidia GPUs
   - Intel CPUs
- Used libraries
   - cuDNN(NVidia GPUs)
   - oneDNN(Intel CPUs)
- Supported data types
   - FP64(cuDNN)
   - FP32
   - FP16
   - INT8
   - UINT8(partially for both)
- Supported cuDNN operations

| Operation | FP64 | FP32 | FP16 | INT16 | INT8 | UINT8 |
|--|--|--|--|--|--|--|
| Convolution | O | O | O | X | O | O |
| Connected | O | O | O | X | O | O |
| Maxpool | O | O | O | X | O | X |
| Avgpool | O | O | O | X | O | X |
| Softmax | O | O | O | X | O | O |
| Route | O | O | O | X | O | O |
| Reorg | O | O | O | X | O | O |
| Element-Wise Addition | O | O | O | X | O | O |
| Upsample | O | O | O | X | O | X |
| Group Convolution | O | O | O | X | O | O |
| Activations | O | O | O | X | O | O |
| Batch Normalization | O | O | O | X | O | X |
| Bias Addition | O | O | O | X | O | X |

- Supported oneDNN operations

| Operation | FP64 | FP32 | FP16 | INT16 | INT8 | UINT8 |
|--|--|--|--|--|--|--|
| Convolution | X | O | O | X | O | O |
| Connected | X | O | O | X | O | O |
| Maxpool | X | O | O | X | O | O |
| Avgpool | X | O | O | X | O | O |
| Softmax | X | O | O | X | O | O |
| Route | X | O | O | X | O | O |
| Reorg | X | O | O | X | O | O |
| Element-Wise Addition | X | O | O | X | O | O |
| Upsample | X | O | O | X | O | O |
| Group Convolution | X | O | O | X | O | O |
| Activations | X | O | O | X | O | O |
| Batch Normalization | X | O | O | X | O | X |
| Bias Addition | X | O | O | X | O | X |

- Tested with
   - VGG16
   - MobileNet
   - ResNet50
   - YOLOv3

## Remaining Objectives

### AIXGraph Operation Support

1. Implement unsupported operations
   - INT16 and UINT8 operations for cuDNN
   - FP64, INT16 and UINT8 operations for oneDNN
2. Create additional operations
   - E.g.) Inverted residual block
3. Re-implement few operations in CUDA
   - Sampling operations (Pixelshuffle, Reorg)
   - Route operation

### Quantization Support

1. Scaling activations
   - Scaling per-tensor or per-channel
   - Scaling symmetrically or asymmetrically
   - Scaling with maximum, percentile, or entropy
2. Quantization simulation data type supports
   - INT16, SINT8, INT8, and hybrid floating points
