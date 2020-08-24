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

from AxfcIRBlock import *
from AxfcGraphWriter import *


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

    ## @var symtab
    # a reference to an IR symbol table

    ## The constructor
    # @param self this object
    # @param symtab a symbol table for referring to an IR node using its name
    def __init__(self, symtab: dict):
        self.root_node = None

        # initialize as an empty list
        self.nodes = list()
        self.blocks = list()
        self.symtab = symtab

    ## This method is used to append the given IR node into the graph
    #
    # @param self this object
    # @param ir_node an IR node to be appended
    def append_node(self, ir_node: AxfcIRNode):

        # update the id of the given IR node
        ir_node.id = len(self.nodes)

        # register the root node of this IR graph
        if len(self.nodes) == 0:
            self.root_node = ir_node

        self.nodes.append(ir_node)

    ## This method is used to append the given IR block into the graph
    #
    # @param self this object
    # @param ir_block an IR block to be appended
    def append_block(self, ir_block: AxfcIRBlock):

        # update the id of the given IR block
        ir_block.id = len(self.blocks)
        self.blocks.append(ir_block)

    ## This method is used to perform the liveness analysis of this graph
    #
    # @param self this object
    # @return error info
    def analyse_liveness(self) -> AxfcError:

        # perform local liveness analysis for each of the blocks
        for block in self._blocks:
            err = block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                return err

        return AxfcError.SUCCESS

    ## This method is used to visualize the IR graph using Sigma js.
    # @param self this object
    # @param file_path a file path to dump out the IR graph
    # @param ignore_ops a list of operations to be ignored
    # @return error info
    def dump_to_file(self, file_path: str, ignore_ops: list) -> AxfcError:

        graph_writer = AxfcGraphWriter()

        # Nested function to ignore edges from a constant node
        def is_ignored(op: str) -> bool:
            try:
                return ignore_ops.index(op) >= 0
            except ValueError as e:
                return False

        # build a AxfcGraph
        for ir_node in self.nodes:

            # ignore some edges
            if is_ignored(ir_node.op):
                continue

            graph_writer.add_node(ir_node)

            for succ in ir_node.succs:
                # ignore some edges
                if is_ignored(succ.op):
                    continue

                graph_writer.add_edge(ir_node.id, succ.id)

        return graph_writer.write_file(file_path)

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
