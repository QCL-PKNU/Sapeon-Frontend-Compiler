# SKT AIX Frontend Compiler #

This README describes the organization and usage of the SKT AIX Frontend Compiler.

### 1. Source Organization ###

#### Common
 
* AxfcFrontendCompiler
* AxfcIRBuilder    
* AxfcIRTranslator
* AxfcMachineDesc
* AxfcGraphWriter
* AxfcIRGraph
* AxfcIRBlock
* AxfcIRNode
* AxfcError
* AxfcMain

#### Tensorflow

* AxfcTFIRBuilder
* AxfcTFIRTranslator

#### SKT-AIX

* aixh_pb2

### 2. Usage ###

    $$ python3 AxfcMain.py [-m] [-i] [-o] [-l] [-g]
    
    -m: Path to a machine description file
    -i: Path to the protocol buffer of a frozen model
    -o: Path to output the generated AIXGraph
    -l: Path to log out (optional)
    -g: Path to dump out an IR graph (optional)

### 3. Contact ###

* Youngsun Han (youngsun@pknu.ac.kr)
* Associate Professor
* Department of Computer Engineering, Pukyong National University