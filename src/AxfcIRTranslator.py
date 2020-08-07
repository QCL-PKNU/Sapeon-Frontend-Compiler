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

import logging

from aixh_pb2 import *
from AxfcError import *
from AxfcIRBlock import *
from AxfcIRGraph import *
from AxfcMachineDesc import *

##
# AxfcIRTranslator
#
class AxfcIRTranslator:

    ## @var _md
    # AIX machine description

    ## The constructor
    def __init__(self, md):
        self._md = md

    ## This method translates IR blocks of the given IR graph into AIXGraphs and
    #  return them.
    #
    # @param self this object
    # @param ir_graph input IR graph
    # @return error info and a list of AIXGraphs
    def emit_aixh_graph(self, ir_graph: AxfcIRGraph) -> {AxfcError, list}:
        logging.info("AxfcIRTranslator:emit_aixh_graph")

        aix_graphs = list()

        for ir_block in ir_graph.blocks:
            err, aix_graph = self.__translate_aixh_block(ir_block)
            if err is AxfcError.SUCCESS:
                aix_graphs.append(aix_graph)
            else:
                return err, None

        return AxfcError.SUCCESS, aix_graphs

    ## This method is used to translate an IR block into an AIXGraph.
    #
    # @param self this object
    # @param ir_block input IR block
    # @return error info and an output AIXGraph
    def __translate_aixh_block(self, ir_block: AxfcIRBlock) -> AIXGraph:
        logging.info("AxfcIRTranslator:__translate_aixh_block - block %d", ir_block.id)
        aix_graph = AIXGraph()
        aix_layer = AIXLayer()

        aix_graph.layer.append(aix_layer)

        return AxfcError.SUCCESS, aix_graph

    def emit_aixh_launcher(self):
        logging.info("AxfcIRTranslator:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass