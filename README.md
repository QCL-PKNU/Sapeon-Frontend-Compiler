
# AIX Frontend Compiler Tutorial
**Version: v2.0**

## Frontend Compiler

### 1. Get source project:
Download the source project named Saepon-Frontend-Compiler

### 2. Install Dependencies
Go to the project directory:

```bash
cd frontend
```
Create a virtual environment
```bash
python3 -m venv venv
```
Activate the virtual environment
```bash
source venv/bin/activate
```
Install the required dependencies
```bash
pip install -r requirements.txt
```




### 3. Compile the model
Deep learning compiler can be compiled into AIXGraph by using the following command line:
```bash
python3 src/AxfcMain.py \
		-m assets.md/tensorflow_sample.md \
		-c assets/calibs/resnet50_v1_imagenet_calib.tbl \
		-i assets/models/resnet50.pb \
		-l assets/logging.log \
		-o aix_graph.out \
		-f binary
```

After compilation, the output AIXGraph (aix_graph.pb) file will be generated in binary format.

How the aix_graph file look like in text format:
```plaintext
layer {
  id: 0
  name: "conv1"
  type: AIX_LAYER_CONVOLUTION
  type: AIX_LAYER_INPUT
  succs: 1
  input {
    dtype: AIX_DATA_FLOAT
    format: AIX_FORMAT_NCHW
    dims: 224
    dims: 224
    dims: 3
    dims: 1
    size: 150528
  }
  output {
    dtype: AIX_DATA_FLOAT
    format: AIX_FORMAT_NCHW
    dims: 112
    dims: 112
    dims: 64
    dims: 1
    size: 802816
  
  filter {
    dtype: AIX_DATA_FLOAT
    format: AIX_FORMAT_NCHW
    dims: 7
    dims: 7
    dims: 3
    dims: 64
    size: 9408
  },
  ...
}
```


## AIXGraph Simulator

### 1. Build and Execute the AIXGraph
Go to the project directory:

```bash
cd simulator
```
Create the Build Directory
```bash
mkdir build
```
Configure the Project with CMake
```bash
cd build
mkdir build
```
Build the Simulator
```bash
cmake ..
make
```
Run the Built Simulator with sample below.
```bash
# Execute aix_graph using built simulator
./simulator --backend cpu \
			--model-path assets/aix_graph.out.0.pb \
			--graph-type aix_graph \
			--infer \
			--image-path assets/cat.jpg \
			--dump-level debug \
			--dump-dir outputs
```
After executing the AIXGraph using the built simulator, the output tensor will be saved in the outputs folder.

##  Automatically Compile DL model and Execute AIXGraph

### 1. Install Dependencies
Make sure the below dependencies are installed.
```bash
pip install numpy  
sudo apt-get install pybind11-dev  
sudo apt-get install python3-dev 
sudo apt-get install libgoogle-glog-dev
```

Install the google probuf to build from the https://github.com/protocolbuffers/protobuf.git
```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.21.12
git submodule update --init --recursive
```



### 2. Using One Line Command Below:

For Tensorflow:
```bash
make all
```
or 
```bash
make all MODEL_TYPE=tensorflow
```

For ONNX:
```bash
make all MODEL_TYPE=onnx
```

For PyTorch:
```bash
make all MODEL_TYPE=pytorch
```
