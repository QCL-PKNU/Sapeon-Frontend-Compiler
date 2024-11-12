# AIXGraph Simulator

DNN inference simulator for AIXGraph. Currently supports inference on NVidia GPUs and Intel CPUs. Supported data types and operations will be explained in more detail [here](#supported-data-types) and [here](#supported-operations).

Current project will soon be more fully documentated.

## Prerequisites

This project was created and tested in Ubuntu 20.04.3 LTS, thus all details will be explained assuming that the environment is Ubuntu.

### AIXGraph Simulator Requirements

- GCC C++ compiler (comes with Ubuntu installation)
```
sudo apt-get install g++
```

- CUDA

[CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)

- cuDNN	

[cuDNN Installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

- CUDA compiler driver (comes with CUDA toolkit installation)	
```
sudo apt-get install nvidia-cuda-toolkit
```

- oneDNN	

[oneDNN Installation](https://oneapi-src.github.io/oneDNN/dev_guide_build.html)

- Protocol buffer:

[Protocol Buffers C++ Installation](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)

- Boost
```
sudo apt-get install libboost-all-dev
```

- OpenCV

[OpenCV Installation for C++](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

### Python Test Script Requirements

- PyTorch

[PyTorch Installation](https://pytorch.org/get-started/locally/)

- OpenCV
```
pip install opencv-python
```

- Protocol buffers
```
pip install google protobuf
```

## Installation

Protobuf compilation:
```
cd aixgraph-simulator/protobuf
protoc --cpp_out=. aixh.proto
mv aixh.pb.h ../include/protobuf/
mv aixh.pb.cc ../src/core/
```

(Optional) OpenCV library configuration (only when shared libraries cannot be loaded):
```
sudo sh -c "echo '/usr/local/lib/' >> /etc/ld.so.conf.d/opencv1.conf"
sudo ldconfig
```

Compilation:
```
git clone https://github.com/SAPEON-Artiference/aixgraph_simulator.git
cd aixgraph-simulator
make all
```

## Simulator Options

-h, --help
- Help option, prints out all simulator options.

-b, --backend
- Backend option, must choose between `cudnn` or `onednn`.
- Must be given at the start of the program.

-p, --protobuf
- Protobuf file path option, requires AIXGraph (.pb) file path.
- Must be given at the start of the program.

-d, --datatype
- Inference data type option, must choose between `fp64`, `fp32`, `fp16`, `sint8` or `uint8`.
- Must be given at the start of the program.

-c, --calibration
- Calibration mode option, must be chosen between `none`, `max`, `percentile`, or `entropy`.
- If `max` or `entropy`, batch size must be given as well.
- If `percentile`, batch size and percentile value must be given as well.
- Must be given at the start of the program.

-s, --save
- Output option, must be chosen from `none`, `bin`, or `csv`.
- If `bin` or `csv`, the output results will be saved into `results` directory.
- Optional, as the default value is `none` given at the start of the program.

### Tutorials

cuDNN single device inference:
```
./simulator -b cudnn -p path/to/aixgraph -d fp32 -c none
```
where `path/to/aixgraph` is an absolute or relative path to the AIXGraph you want to run.

oneDNN single device inference:
```
./simulator -b onednn -p path/to/aixgraph -d fp32 -c none
```
where `path/to/aixgraph` is an absolute or relative path to the AIXGraph you want to run.

Saving output results:
```
./simulator -b backend -p path/to/aixgraph -d fp32 -c none -s csv
```
where `backend` is a valid backend option of your choice, and `path/to/aixgraph` is an absolute or relative path to the AIXGraph you want to run.

Quantize AIXGraph with entropy calibration:
```
./simulator -b backend -p path/to/aixgraph -d fp32 -c entropy 1000
```
where `backend` is a valid backend option of your choice, and `path/to/aixgraph` is an absolute or relative path to the AIXGraph you want to run.

## Supported Data Types

On cuDNN:
- FP64
- FP32
- SINT8
- UINT8
   - Batch normalization and pooling operations are not supported in UINT8.

On oneDNN:
- FP32
- SINT8
- UINT8
   - Batch normalization operation is not supported in UINT8.

## Supported Operations

- Activation
   - Sigmoid
   - ReLU
   - LeakyReLU
- Avgpool
- Batch normalization (without bias addition)
- Bias addition
- Convolution (without bias addition)
- Connected
- Element-wise add
- Group convolution
- Maxpool
- Pixelshuffle
- Reorg
- Route
- Softmax
- Upsample

## Implementation Detail

Templated factory pattern:
- Used templated factory pattern for various classes to avoid writing switch-case or if-else statements.
- Static member variable `registered` will be initialized first at compile time, which registers its `create()` to respective factories.
   - The `create()` function is used to create the necessary class.
   - Uses the smart pointer when registerting to factories.
   - Must be careful of static initialization fiasco.

Modularized operation implementations:
 - Operations implemented in cuDNN and oneDNN have similar outlooks during inference, thus similar functions were modularized into bigger function for readability.

### AIXGraph Simulator Flow

1. `checkArguments()` in simulator.cpp
   - Checks validity all arguments given to this simulator
2. `checkSupport()` in simulator.cpp
   - Checks graph's all layer operations are available for inference in this simulator
3. `inference()` in Backend class
   1. `runInference()` with `minmax_found` set as `false`
      - Runs inference for counting
	  - Will not find maximum values per channels of every output if calibration is set to `none`
   2. `runInference()` with `minmax_found` set as `true`
      - Runs inference for collecting values
	  - Will not collect values of every output if calibration is set to `none`
   3. `runInference()` with Quantization class set
      - Will not run quantization if it is set to `none`
4. `forward()` in CudnnOperation/OnednnOperation class
   1. `CalculateDimensions()`
      - Calculates output dimension
   2. `CreateDescriptors()`/`CreatePrimitives()`
      - Creates memory descriptors or primitives (`CreatePrimitives()` are only for oneDNN)
   3. `AllocateMemory()`
      - Allocate and transfer input data (activation, weight, bias, etc) to designated memory region
   4. `OperationForward()`
      - Forwarding operation
   5. `GetOutput()`
      - Transfer output data to host memory
   6. `DeAllocateMemory()`
      - Deallocate used host memory
