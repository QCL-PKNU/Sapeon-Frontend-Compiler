#######################################################################
#   AxfcIRTranslator
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import enum
import logging

from aixh_pb2 import *
from AxfcError import *
from AxfcIRBlock import *
from AxfcIRGraph import *
from AxfcMachineDesc import *

#######################################################################
# AxfcIRTranslator class
#######################################################################
class AxfcIRTranslator:

    ## AIX data type
    class AIXDataType(enum.Enum):
        AIX_DATA_FLOAT = 0
        AIX_DATA_DOUBLE = 1
        AIX_DATA_HALF = 2
        AIX_DATA_UINT8 = 3
        AIX_DATA_SINT8 = 4
        AIX_DATA_SINT16 = 5

    ## AIX tensor format
    class AIXTensorFormat(enum.Enum):
        AIX_FORMAT_NCHW = 0
        AIX_FORMAT_NHWC = 1
        AIX_FORMAT_NWHC = 2
        AIX_FORMAT_VECTOR = 3

    ## @var _md
    # AIX machine description

    ## The constructor
    def __init__(self, md):
        self._md = md

    def _emit_aixh_node(self, ir_node: AxfcIRNode) -> {AxfcError, AIXLayer}:
        return NotImplementedError()

    ## This method translates IR blocks of the given IR graph into AIXGraphs and
    #  return them.
    # @param self this object
    # @param ir_graph input IR graph
    # @return error info and a list of AIXGraphs
    def emit_aixh_graphs(self, ir_graph: AxfcIRGraph) -> {AxfcError, list}:
        logging.info("AxfcIRTranslator:emit_aixh_graph")

        # create a new list of AIX graphs to output
        aix_graphs = list()

        # translate all the blocks into AIX graphs
        for ir_block in ir_graph.blocks:
            err, aix_graph = self.__emit_aixh_block(ir_block)
            if err is AxfcError.SUCCESS:
                aix_graphs.append(aix_graph)
            else:
                return err, None

        return AxfcError.SUCCESS, aix_graphs

    ## This method is used to translate an IR block into an AIXGraph.
    # @param self this object
    # @param ir_block input IR block
    # @return error info and an output AIXGraph
    def __emit_aixh_block(self, ir_block: AxfcIRBlock) -> {AxfcError, AIXGraph}:
        logging.info("AxfcIRTranslator:__emit_aixh_block - block %d", ir_block.id)

        # create a new AIX graph to output
        aix_graph = AIXGraph()

        # translate all the nodes into AIX layers
        for ir_node in ir_block.nodes:
            err, aix_layer = self._emit_aixh_node(ir_node)
            if err is AxfcError.SUCCESS:
                aix_graph.layer.append(aix_layer)
            else:
                return err, None

        # CHKME - YOUNGSUN (2020.08.07)
        # need to configure input_layers and output_layers

        return AxfcError.SUCCESS, aix_graph

    def emit_aixh_launcher(self):
        logging.info("AxfcIRTranslator:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass