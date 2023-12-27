# SKT AIX Frontend Compiler

This README describes the organization and usage of the SKT AIX Frontend Compiler.

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