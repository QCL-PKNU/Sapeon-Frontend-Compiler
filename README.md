# SKT AIX Frontend Compiler #

This README describes the organization and usage of the SKT AIX Frontend Compiler.

## <strong> Source Organization </strong> ##

### <strong> Common </strong> ###

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

### <strong> Util </strong> ###

* AxfcAIXLayerView
* AxfcCustomGraph
* AxfcTFGraphUtil
* AxfcUtil

### <strong> Tensorflow </strong> ###

* AxfcTFIRBuilder
* AxfcTFIRTranslator

### <strong> SKT-AIX </strong> ###

* aixh_pb2

## <strong> Prerequisites </strong> ##
* Installation
  > pip3 install -r requirements.txt

## <strong> Usage </strong> ##  

Our frontend compiler currently provides 2 ways for the executing, by using Makefile or python3 command line.
### <strong> Using Python3 Command Line </strong> ###
To use the python3 command line, we have to pass the required arguments listed below.

<strong>Required Arguments</strong>
    
    -m: path to a machine description file 
    -i: path to the protocol buffer of a frozen model
    -k: path to the kernel (custom operation kernerl *.so) file

<strong> Optional Arguments </strong>

    -c: Path to the calibration data of a frozen model (optional)
    -o: Path to output the generated AIXGraph (optional)
    -l: Path to log out (optional)
    -g: Path to dump out an IR graph (optional)
    -f: Configure output for aix graph format between 'binary' and 'text' (optional, default is binary)
  Note:

* For -f argument, we recommend to use binary format as it is much faster for dumping the aix graph.

<strong> Example </strong>

1. On terminal, go to aix frontend compiler directory
    ```    
    $ cd skt-aix-frontend-compiler
    ```
2. Run aix compiler
   ```
   $ python3 src/AxfcMain.py -m=tst/model_description.md -i=tst/model_name.pb -k=tst/custom_op_kernel.so -f=binary
   ```
### <strong> Using Makefile: </strong> ###
To use the makefile, please follow the following steps below:
1. Configure makefile, go to edit Makefile at path ``skt-aix-frontend-compiler/Makefile``
2. Fill in the required parameter belows:
   ```
   MODEL= ./tst/model_name.pb 
   MD= ./tst/model_description.md
   KERNEL= ./tst/custom_op_kernel.so
   ```
3. On terminal, go to aix frontend compiler directory:
    ```    
    $ cd skt-aix-frontend-compiler
    ```
4. Run Makefile
    ```
    $ make all
    ```

## <strong> Contact </strong> ##

* Youngsun Han (youngsun@pknu.ac.kr)
  * Associate Professor
  * Department of Computer Engineering, Pukyong National University
* Sengthai Heng (sengthai37@gmail.com)
  * Graduated Student
  * Department of AI Convergence, Pukong National University
* Leanghok Hour (leanghok@pukyong.ac.kr)
  * Graduated Student
  * Department of AI Convergence, Pukyong National University
