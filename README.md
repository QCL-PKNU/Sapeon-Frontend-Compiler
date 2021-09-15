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

### **SKT-AIX**

* aixh_pb2

## **Prerequisites**
* Installation
  ```
  $ pip3 install -r requirements.txt
  ```
* Launcher: please kindly refer to aix-launcher project prerequisites, please make sure to create the virtual environment and install library there. 

## **Usage** 

Our frontend compiler currently provides 2 ways for the executing, by using Makefile or python3 command line.
### **Using Python3 Command Line**
To use the python3 command line, we have to pass the required arguments listed below.

**Required Arguments**
    
    -m: path to a machine description file 
    -i: path to the protocol buffer of a frozen model
    -k: path to the skt-launcher project

 **Optional Arguments**

    -c: Path to the calibration data of a frozen model (optional)
    -o: Path to output the generated AIXGraph (optional)
    -l: Path to log out (optional)
    -g: Path to dump out an IR graph (optional)
    -f: Configure output for aix graph format between 'binary' and 'text' (optional, default is binary)
  Note:

* For -k argument, we have updated from using the custom_kernel.so to launcher directory path.
* For -f argument, we recommend to use binary format as it is much faster for dumping the aix graph.
 
 **Example**

1. On terminal, go to aix frontend compiler directory
    ```    
    $ cd skt-aix-frontend-compiler
    ```
2. Run aix compiler
   ```
   $ python3 src/AxfcMain.py -m=tst/model_description.md -i=tst/model_name.pb -k=/home/{username}/Documents/aix/skt-aix-launcher -f=binary
   ```
### **Using Makefile:**
To use the makefile, please follow the following steps below:

1. Configure makefile, go to edit Makefile at path ``skt-aix-frontend-compiler/Makefile``


2. Fill in the required parameter belows:
   ```
   MODEL= ./tst/model_name.pb 
   MD= ./tst/model_description.md
   KERNEL= /home/{username}/Documents/aix_pro/skt-aix-launcher
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
