Here's the updated README with the explanation of the output file generated after running `make run`:

---

# Sapeon Frontend Compiler

This README describes the organization and usage of the Sapeon Frontend Compiler.

## **Release Note v3.0 (UAT)**

## **Source Organization**

### **Common**

- `AxfcFrontendCompiler`
- `AxfcIRBuilder`
- `AxfcIRTranslator`
- `AxfcMachineDesc`
- `AxfcGraphWriter`
- `AxfcLauncherWriter`
- `AxfcLauncher`
- `AxfcIRGraph`
- `AxfcIRBlock`
- `AxfcIRNode`
- `AxfcError`
- `AxfcMain`

### **Util**

- `AxfcAIXLayerView`
- `AxfcCustomGraph`
- `AxfcTFGraphUtil`
- `AxfcUtil`

### **TensorFlow**

- `AxfcTFIRBuilder`
- `AxfcTFIRTranslator`
- `AxfcTFWriter`

### **ONNX**

- `AxfcONNXBuilder`
- `AxfcONNXIRTranslator`
- `AxfcONNXWriter`

### **PyTorch**

- `AxfcPTBuilder`
- `AxfcPTTranslator`
- `AxfcPTWriter`

### **SKT-Sapeon**

- `aixh_pb2`

## **Prerequisites**

Manaully Installing Dependency (Ubuntu 20.04 ~)

to avoid module conflicts, we reconnect to create a virtual ENV by following the following command:

```bash
# Create
python3.9 -m venv venv

# Activate
source venv/bin/activate

# Install
pip install -r requirements.txt
```

## Model Compilation

The Sapeon Frontend Compiler converted DL model into **AIXGraph** format. Follow the steps below for different frameworks:

### Step 1: Enter the Scripts Directory

```bash
cd scripts
```

### Step 2: Update File Permissions

Ensure the shell script file is executable:

```bash
chmod +x _file_name_.sh
```

### Step 3: Compile Models

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

### Step 4: Output AIXGraph Format

After running the compilation, the output **AIXGraph** file (e.g., `aix_graph.0.aixt`) will initially be generated in text format. To obtain the binary format, convert it to `aix_graph.0.aixb` by setting `GRAPH_FORMAT=binary`.

## **Cleaning Up Files**

To clean up build artifacts and temporary files generated during execution (such as `__pycache__`, and `venv`), run the `make clean` command:

```bash
$ make clean
```

## **Contact**

- **Youngsun Han (youngsun@pknu.ac.kr)**

  - Associate Professor
  - Department of Computer Engineering, Pukyong National University

- **Sengthai Heng (sengthai37@gmail.com)**

  - Graduate Student
  - Department of AI Convergence, Pukyong National University

- **Leanghok Hour (leanghok@pukyong.ac.kr)**

  - Graduate Student
  - Department of AI Convergence, Pukyong National University

- **Myeongseong Go (gms3089@pukyong.ac.kr)**

  - Graduate Student
  - Department of AI Convergence, Pukyong National University

- **Kimsay Pov (povkimsay@gmail.com)**
  - Graduate Student
  - Department of AI Convergence, Pukyong National University

---
