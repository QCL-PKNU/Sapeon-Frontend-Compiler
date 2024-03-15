# SKT AIX Frontend Compiler

This README describes the organization and usage of the SKT AIX Frontend Compiler.

## **Release Note v2.0 (UAT)**

### **New Updates**

This release includes features:

1. Enable to compile PyTorch-to-AIXGraph  

    We have implemented parser and translator for PyTorch model, the following files were added for the purpose of this implementation:  

    * src/AxfcPyTorchIRBuilder: parser for PyTorch graph
    * src/AxfcPyTorchIRTranslator: translator for PyTorch graph   
    * src/AxfcPyTorchWriter: custom model generator for PyTorch model 

2. Enable to generate PyTorch custom graph.  

    The custom model replaces AIXGraph with arbitarily operator (custom domain) on PyTorch model.
    
    ![Scheme](doc/images/ex_torch_model.png)
        
    * Example of an PyTorch model and compiled PyTorch model  

3. Support multiple AIXGraphs or/and AIXOps for TensorFlow, ONNX, and PyTorch

    Like shown in No.2 multiple AIXGraphs can be generated from a model and supported model includes, TensorFlow, ONNX, and PyTorch.

### **Open Issues**

This release contains the following issue that needs to be addressed:

1. Generated AIXGraph only contains input information (type, dims, size) but does not contain input value (fval) for AIX_LAYER_INPUT.    

    Generated AIXGraph only contains input information:

      * Generated AIXGraph get the input from the previous node in the model so it only contains input information like type, dims and size.  
      It doesn’t contain the input value (fval) for the AIX_LAYER_INPUT node. Due to this reason, when testing with AIXGraph simulator, the input value (fval) needs to be manually inserted.
      
      ![Scheme](doc/images/ex_gen_aixgraph.png)

2. TensorFlow custom model cannot be generated due to the required re-implementation of the custom operation kernel to change support from Darknet to AIXGraph simulator.

    TensorFlow custom model cannot be generated:

    * In this release, the compiler can generate AIXGraph, includes multiple AIXGraph from a model.  
     However, the custom model that contains AIXOp cannot be generated. Because to generate a custom operation like AIXOp on TensorFlow, it is required to have a custom kernel for that specific operation.  
     Previously, we implemented the custom kernel using Darknet but now we have shifted to use on AIXGraph simulator. Thus, the custom kernel needs to be rewritten to support on AIXGraph simulator.

## **Source Organization**

### **Common**

* AxfcFrontendCompiler
* AxfcIRBuilder
* AxfcIRTranslator
* AxfcMachineDesc
* AxfcGraphWriter
* AxfcLauncherWriter
* AxfcLauncher  
* AxfcIRGraph
* AxfcIRBlock
* AxfcIRNode
* AxfcError
* AxfcMain

### **Util**

* AxfcAIXLayerView
* AxfcCustomGraph
* AxfcTFGraphUtil
* AxfcUtil

### **Tensorflow**

* AxfcTFIRBuilder
* AxfcTFIRTranslator
* AxfcTFWriter

### **ONNX**
* AxfcONNXBuilder
* AxfcONNXIRTranslator
* AxfcONNXWriter

### **PyTorch**
* AxfcPTBuilder
* AxfcPTTranslator
* AxfcPTWriter

### **SKT-AIX**

* aixh_pb2

## **Prerequisites**
Install Dependencies (Ubuntu 20.04~)

To prohibit the module confilit, recomend creating a virtual environment.

```
$ python3 -m venv {virtual_env_name}

1.Ubuntu
  $ cd {virtual_env_name}
  $ source bin/activate

2.Window
  $ cd {virtual_env_name}/Scripts
  $ activate.bat
```

And install the requirement packages.
```
$ pip3 install -r requirements.txt
```
If the `onnx_graphsurgeon` package is not be installed successfully, please check the latest version and installed it in latest version.

```
pip3 install onnx_graphsurgeon={latest_version}
```
## **Usage** 
Our frontend compiler currently provides 2 ways for the executing, by using Makefile or python3 command line.

### **Using Python3 Command Line**
To use the python3 command line, we have to pass the required arguments listed below.

**Required Arguments**
    
    -m: path to a machine description file 
    -i: path to the protocol buffer of a frozen model

 **Optional Arguments**

    -c: Path to the calibration data of a frozen model (optional)
    -o: Path to output the generated AIXGraph (optional)
    -l: Path to log out (optional)
    -g: Path to dump out an IR graph (optional)
    -f: Configure output for aix graph format between 'binary' and 'text' (optional, default is binary)
  Note:

* For -f argument, we recommend to use binary format as it is much faster for dumping the aix graph.
 
 **Example**

1. On terminal, go to aix frontend compiler directory
    ```    
    $ cd skt-aix-frontend-compiler
    ```
2. Run aix compiler
   ```
   $ python3 src/AxfcMain.py -m=tst/machine_description.md -i=tst/model_name.pb -f=text
   ```

   Note: you can find sample machine description file for ONNX model (onnx_sample.md) and TensorFlow model (tf_sample.md) in the 'skt-aix-frontend-compiler/tst' directory.

### **Using Makefile:**
To use the makefile, please follow the following steps below:

1. Configure makefile, go to edit Makefile at path ``skt-aix-frontend-compiler/Makefile``


2. Fill in the required parameter belows:
   ```
   MODEL= ./tst/model_name.pb 
   MD= ./tst/model_description.md
   ```
3. On terminal, go to aix frontend compiler directory:
    ```    
    $ cd skt-aix-frontend-compiler
    ```
4. Run Makefile
    ```
    $ make all
    ```

## **Contact**

Youngsun Han (youngsun@pknu.ac.kr)

* Associate Professor
* Department of Computer Engineering, Pukyong National University

Sengthai Heng (sengthai37@gmail.com)

* Graduated Student
* Department of AI Convergence, Pukong National University

Leanghok Hour (leanghok@pukyong.ac.kr)

* Graduated Student
* Department of AI Convergence, Pukyong National University

Sanghyeon Lee (sanghyeon@pukyong.ac.kr)

* Graduated Student
* Department of AI Convergence, Pukyong National University

Myeongseong Go (gms3089@pukyong.ac.kr)

* Graduated Student
* Department of AI Convergence, Pukyong National University

Kimsay Pov (povkimsay@gmail.com)

* Graduated Student
* Department of AI Convergence, Pukyong National University