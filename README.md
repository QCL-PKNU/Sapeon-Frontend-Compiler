# AIX Frontend Compiler and Simulator Tutorial  
**Version: v2.0**  

This guide provides step-by-step instructions for compiling and executing deep learning models using the AIX Frontend Compiler and Simulator.  

---

## Frontend Compiler  

### 1. Get the Source Project  
Clone the repository for the **Saepon-Frontend-Compiler** project.  

---

### 2. Install Dependencies  
Navigate to the project directory and create a virtual environment for the frontend:  

```bash  
make frontend/venv  
```  

---

### 3. Compile the Model  
The AIX Frontend Compiler converts deep learning models into **AIXGraph** format. Follow the steps below for different frameworks:  

#### Step 1: Enter the Scripts Directory  
```bash  
cd scripts  
```  

#### Step 2: Update File Permissions  
Ensure the shell script file is executable:  
```bash  
chmod +x _file_name_.sh  
```  

#### Step 3: Compile Models  

**Compile a TensorFlow Model (e.g., ResNet50)**:  
```bash  
./c_tf_resnet50.sh  
```  

**Compile an ONNX Model (e.g., ResNet50)**:  
```bash  
./c_onnx_resnet50.sh  
```  

**Compile a PyTorch Model (e.g., ResNet50)**:  
**Note**: For PyTorch, you must pass the `INPUT_SHAPE` as a parameter to generate a static AIXGraph.  
```bash  
./c_pt_resnet50.sh  
```  

#### Step 4: Output AIXGraph Format
After running the compilation, the output **AIXGraph** file (e.g., `aix_graph.0.aixt`) will initially be generated in text format. To obtain the binary format, convert it to `aix_graph.0.aixb` by setting `GRAPH_FORMAT=binary`.

---

### Example AIXGraph File in Text Format  
The AIXGraph file in text format contains information such as layer properties, input/output tensors, dimensions, and more. Here's an example:  

```plaintext  
layer {  
  id: 0  
  name: "conv1"  
  type: AIX_LAYER_CONVOLUTION  
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
  }  
  filter {  
    dtype: AIX_DATA_FLOAT  
    format: AIX_FORMAT_NCHW  
    dims: 7  
    dims: 7  
    dims: 3  
    dims: 64  
    size: 9408  
    ...  
  }  
  ...  
}  
```  

---

## AIXGraph Simulator  

### 1. Build and Execute the AIXGraph  

#### Step 1: Navigate to the Simulator Directory  
```bash  
cd simulator  
```  

#### Step 2: Create a Build Directory  
```bash  
mkdir build && cd build  
```  

#### Step 3: Configure the Project with CMake  
```bash  
cmake ..  
make  
```  

#### Step 4: Run the Built Simulator  
Use the following command to execute the compiled AIXGraph:  
```bash  
./simulator --backend cpu \  
            --model-path assets/aix_graph.out.0.pb \  
            --graph-type aix_graph \  
            --infer \  
            --image-path assets/cat.jpg \  
            --dump-level debug \  
            --dump-dir outputs  
```  

After execution, the output tensor will be saved in the `outputs` folder.  

---

## Automating Model Compilation and Execution  

### 1. Install Dependencies  
Ensure the following dependencies are installed:  
```bash  
pip install numpy  
sudo apt-get install pybind11-dev  
sudo apt-get install python3-dev  
sudo apt-get install libgoogle-glog-dev  
```  

#### Install Protocol Buffers  
Clone and build Protocol Buffers from GitHub:  
```bash  
git clone https://github.com/protocolbuffers/protobuf.git  
cd protobuf  
git checkout v3.21.12  
git submodule update --init --recursive  
```  