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

    """
    Represents an IR graph with nodes, blocks, and a root node, alongside a symbol table.
    
    Attributes:
        nodes (List): A list of nodes in this graph.
        blocks (List): A list of blocks contained in this graph.
        root_node: The output root node of this graph, if any.
        symtab (Dict): A reference to an IR symbol table for node names.
    """


    def __init__(self, symtab: dict):
        self.root_node = None
        self.nodes = list()
        self.blocks = list()
        self.symtab = symtab

    
    def append_node(self, ir_node: AxfcIRNode):
        """Appends the given IR node into the graph.

        Args:
            ir_node: An IR node to be appended.
        """

        # update the id of the given IR node
        ir_node.id = len(self.nodes)

        # register the root node of this IR graph
        if len(self.nodes) == 0:
            self.root_node = ir_node

        self.nodes.append(ir_node)


    def append_block(self, ir_block: AxfcIRBlock):
        """Appends the given IR block into the graph.

        Args:
            ir_block: An IR block to be appended.
        """

        # update the id of the given IR block
        ir_block.id = len(self.blocks)
        self.blocks.append(ir_block)


    def analyse_liveness(self) -> AxfcError:
        """Performs liveness analysis on the graph.

        Returns:
            An error code indicating the success or failure of the analysis.
        """

        # perform local liveness analysis for each of the blocks
        for block in self._blocks:
            err = block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                return err

        return AxfcError.SUCCESS


    def dump_to_file(self, file_path: str, ignore_ops: list) -> AxfcError:
        """Visualizes the IR graph using Sigma js and dumps it to a specified file.

        This method iterates over all nodes and their successors in the graph, adding them
        to the graph representation unless they are specified to be ignored.

        Args:
            file_path: The path where the IR graph visualization will be saved.
            ignore_ops: A list of operation names to be ignored during the graph construction.

        Returns:
            An error code indicating the success or failure of the operation.
        """
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


    def __str__(self):
        """Returns a string representation of the AIXIRGraph instance."""

        blocks_str = '\n'.join(str(block) for block in self.blocks)
        nodes_str = '\n'.join(str(node) for node in self.nodes)

        return f">> IRGraph: \n\n>> Blocks of IRGraph: \n{blocks_str}\n\n>> Nodes of IRGraph: \n{nodes_str}\n"
