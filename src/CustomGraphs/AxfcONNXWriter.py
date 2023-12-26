#######################################################################
#   AxfcIRTranslator
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#      Hour Leanghok (hourleanghok@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################


import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs

from AxfcIRGraph import *
from AxfcMachineDesc import *


#######################################################################
# AxfcONNXWriter class
#######################################################################


class AxfcONNXWriter:

    # @var __frozen_model_path
    # the path of aix graph

    # @var __aix_graph_path
    # the path of aix graph

    # @var __kernel_op_path
    # the custom kernel library (*.so) path

    # @var __last_inout_tensors
    # the last unsupported subgraphs.
    # it refers to the layers that not supported by AIX hardware/simulator.

    ## @var __md
    # AIX machine description

    ## @var __ir_graph
    # an AIXIR graph that will be used for writing the launcher

    ## The constructor
    # def __init__(self, md: AxfcMachineDesc, ir_graph: AxfcIRGraph):
    #     self.__md = md
    #     self.__ir_graph = ir_graph

    ## The constructor
    def __init__(self, frozen_model_path: str,
                 aix_graph_path: str,
                 kernel_op_path: str,
                 ir_graph: AxfcIRGraph,
                 md:AxfcMachineDesc):
        self.__ir_graph = ir_graph
        self.__frozen_model_path = frozen_model_path
        self.__aix_graph_path = aix_graph_path
        self.__kernel_op_path = kernel_op_path
        self.__md = md

    # This method is used to build the custom graph
    # @return custom_graph the custom graph from the custom kernel library
    def get_custom_graph(self):
        onnx_model = onnx.load(self.__frozen_model_path)

        inferred_model = shape_inference.infer_shapes(onnx_model)

        gs_graph = gs.import_onnx(inferred_model)

        gs_tensors = gs_graph.tensors()

        #Custom op inline function for graph surgeon
        @gs.Graph.register()
        def replace_with_aixop(self, inputs, outputs, path):
            for inp in inputs:
                inp.outputs.clear()

            for out in outputs:
                out.inputs.clear()
            
            return self.layer(name="AixOp", 
                                op="AixOp", 
                                inputs= inputs,
                                outputs = outputs,
                                attrs=dict(aix_graph_path=path))


        for count, block in enumerate(self.__ir_graph.blocks):

            if not block.is_aixh_support:
                continue
            
            inputs = [gs_tensors[node.name] for node in block.input_nodes]
            outputs = [gs_tensors[node.name] for node in block.output_nodes]

            gs_graph.replace_with_aixop(inputs, outputs, self.__aix_graph_path+str(count))
            # Since, after replace, the nodes lose their connection, need to be topologically sorted
            gs_graph.cleanup().toposort()

        save_path = self.__frozen_model_path.split(".onnx")[0] + "_custom.onnx" 
       
        onnx.save(gs.export_onnx(gs_graph), save_path)

        return AxfcError.SUCCESS, save_path


    ## For debugging
    def __str__(self):
        pass
