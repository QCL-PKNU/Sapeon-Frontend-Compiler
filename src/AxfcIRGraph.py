#######################################################################
#   AxfcIRGraph
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

from AxfcError import *
from AxfcIRNode import *
from AxfcIRBlock import *

#######################################################################
# AxfcIRGraph class
#######################################################################
class AxfcIRGraph:

    ## @var nodes
    # a list of nodes consisting this graph

    ## @var blocks
    # a list of blocks that are contained this graph

    ## @var root_node
    # output root node of this graph

    ## The constructor
    def __init__(self):
        self.root_node = None

        # initialize as an empty list
        self.nodes = list()
        self.blocks = list()

    ## This method is used to append the given IR node into the graph
    # @param self this object
    # @param node an IR node to be appended
    def append_node(self, ir_node: AxfcIRNode):

        # update the id of the given IR node
        ir_node.id = len(self.nodes)

        # register the root node of this IR graph
        if ir_node.is_root:
            self.root_node = ir_node
        else:
            self.root_node = None

        self.nodes.append(ir_node)

    ## This method is used to append the given IR block into the graph
    # @param self this object
    # @param node an IR block to be appended
    def append_block(self, ir_block: AxfcIRBlock):

        # update the id of the given IR block
        ir_block.id = len(self.blocks)

        self.blocks.append(ir_block)

    ## This method is used to perform the liveness analysis of this graph
    # @param self this object
    # @return error info
    def analyse_liveness(self) -> AxfcError:

        # perform local liveness analysis for each of the blocks
        for block in self._blocks:
            err = block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                return err

        return AxfcError.SUCCESS

    ## For debugging
    def __str__(self):
        str_buf = ">> IRGraph: \n\n"

        str_buf += ">> Blocks of IRGraph: \n"
        for block in self.blocks:
            str_buf += str(block) + "\n"

        str_buf += ">> Nodes of IRGraph: \n"
        for node in self.nodes:
            str_buf += str(node) + "\n"

        return str_buf