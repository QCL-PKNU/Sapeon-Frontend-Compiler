#######################################################################
#   AxfcIRBlock
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from AxfcError import *


#######################################################################
# AxfcIRBlock class
#######################################################################

class AxfcIRBlock:
    """Represents an IR block with various attributes.

    Attributes:
        id (int): Block ID.
        node`s (list): A list of nodes that make up this block.
        live_in (set): A set of live-in node IDs.
        live_out (set): A set of live-out node IDs.
        is_aixh_support (bool): Indicates whether this node can be executed in a hardware manner.
        aixh_profit (int): Specifies the profit to be obtained by using AIXH.
        aix_graph: An AIX graph emitted from this IR block.
        input_nodes (list): Input node of the IR block; can be multiple.
        output_node: Outpu`t node of the IR block; must be only one.
    """
    
    def __init__(self):
        self.id = 0
        self.nodes = list()
        self.live_in = set()
        self.live_out = set()
        self.is_aixh_support = False
        self.aixh_profit = 0
        self.aix_graph = None
        self.input_nodes = list() #input node of the ir block can be multiple
        self.output_node = None #output node of the ir block must be only one

    ## This method is used to perform the local liveness analysis in the scope of an IR block.
    #  We employ a simple heuristic scheme to find live-ins and live-outs of a block without
    #  global liveness analysis on the entire IR graph.
    #
    # @param self this object
    # @return error info
    def analyse_liveness(self) -> AxfcError:
        # logging.info("AxfcIRBlock:analyse_liveness")

        # check if the block is ready to be analyzed
        if self.nodes is None:
            return AxfcError.EMPTY_IR_BLOCK

        # find the input and output nodes of this block because
        # we may use the information for the liveness analysis in the future
        self.__analyse_inout()

        uses = set()
        defs = set()

        # find use and def nodes of this block
        for node in self.nodes:
            # def
            defs.add(node.id)
            # use
            for pred in node.preds:
                uses.add(pred.id)

            # compute the live-out from this block
            for succ in node.succs:
                # strict substitution -> if self != succ.block_ref:
                if not succ.is_aixh_support:
                    self.live_out.add(node.id)

        # compute the live-in into this block
        self.live_in = uses - defs

        return AxfcError.SUCCESS

    
    def __analyse_inout(self) -> AxfcError:
        """Find the input and output nodes of this block.

        Returns:
            AxfcError: Error code indicating the success or failure of the operation.
        """
        # logging.info("AxfcIRBlock:analyse_inout")

        # check if the block is ready to be analyzed
        if self.nodes is None:
            return AxfcError.EMPTY_IR_BLOCK

        return AxfcError.SUCCESS


    def analyze_profit(self) -> AxfcError:
        """
        Calculates the potential profit of accelerating this block with hardware support. 
        Profit is defined as the cumulative benefit of executing nodes within this block on hardware.

        Returns:
            AxfcError: Error code indicating the success or failure of the operation.
        """
        # logging.info("AxfcIRBlock:analyze_profit")

        # total profit
        profit = 0

        # check if the block is ready to be analyzed
        if self.nodes is None:
            return AxfcError.EMPTY_IR_BLOCK, -1

        # check if the block is aixh-supported
        if self.is_aixh_support:
            return AxfcError.NOT_AIXH_SUPPORT, -1

        # accumulate the profit of each node in this block
        for node in self.nodes:
            #skip if node is a Const or Pad
            if node.op not in ["Const", "Pad"]:
                profit += node.analyze_profit()

        self.aixh_profit = profit
        return AxfcError.SUCCESS


    def __str__(self):
        """Returns a string representation of the AIXIRBlock instance."""

        nodes_str = ", ".join(
            f"{'*' if node.is_input else ''}{'+' if node.is_output else ''}{node.op}({node.id})"
            for node in self.nodes
        )

        live_in_str = ", ".join(str(live_in) for live_in in self.live_in)
        live_out_str = ", ".join(str(live_out) for live_out in self.live_out)

        return (
            f">> IR Block: {self.id} ( # of nodes: {len(self.nodes)})\n"
            f">> Nodes: [{nodes_str}]\n"
            f">> Live-in: [{live_in_str}]\n"
            f">> Live-out: [{live_out_str}]\n"
            f">> Attributes [aixh_profit: {self.aixh_profit}, aixh_support: {self.is_aixh_support}]\n"
        )