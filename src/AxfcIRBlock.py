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

    ## @var id
    # block ID

    ## @var nodes
    # a list of nodes that make up this block

    ## @var live_in
    # a list of live-in node IDs

    ## @var live_out
    # a list of live-out node IDs

    ## @var is_aixh_support
    # indicate whether this node can be executed in hardware-manner

    ## @var aixh_profit
    # specify the profit to be obtained by using AIXH

    ## @var aix_graph
    # an AIX graph emitted from this IR block

    ## The constructor
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

    ## This method is used to find the input and output nodes of this block.
    #
    # @param self this object
    # @return error info.
    def __analyse_inout(self) -> AxfcError:
        # logging.info("AxfcIRBlock:analyse_inout")

        # check if the block is ready to be analyzed
        if self.nodes is None:
            return AxfcError.EMPTY_IR_BLOCK

        # find input nodes
        for ir_node in self.nodes:
            num_aixh_support = 0

            # count the number of predecessors in this block
            for pred in ir_node.preds:
                if pred.is_aixh_support:
                    num_aixh_support += 1

            # specify the node is an input node if there is no predecessors
            if num_aixh_support == 0:
                ir_node.is_input = True
            else:
                ir_node.is_input = False

        # find output nodes
        for ir_node in self.nodes:
            num_aixh_support = 0

            # count the number of successors in this block
            for succ in ir_node.succs:
                if succ.is_aixh_support:
                    num_aixh_support += 1

            # specify the node is an output node if there is no successors
            if num_aixh_support == 0:
                ir_node.is_output = True
            else:
                ir_node.is_output = False

        return AxfcError.SUCCESS

    ## This method is used to calculate the profit that we can achieve by
    #  accelerating this block in hardware-manner.
    #
    # @param self this object
    # @return error info
    def analyze_profit(self) -> AxfcError:
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

    ## For debugging
    def __str__(self):
        str_buf = ">> IR Block: " + str(self.id)
        str_buf += "( # of nodes: " + str(len(self.nodes)) + ")\n"

        str_buf += ">> Nodes: ["
        for node in self.nodes:
            if node.is_input:
                str_buf += "*"
            if node.is_output:
                str_buf += "+"
            str_buf += str(node.op) + "(" + str(node.id) + "), "
        str_buf += "]\n"

        str_buf += ">> Live-in: ["
        for live_in in self.live_in:
            str_buf += str(live_in) + ", "
        str_buf += "]\n"

        str_buf += ">> Live-out: ["
        for live_out in self.live_out:
            str_buf += str(live_out) + ", "
        str_buf += "]\n"

        str_buf += ">> Attributes [aixh_profit: " + str(self.aixh_profit)
        str_buf += ", aixh_support: " + str(self.is_aixh_support) + "]\n"

        return str_buf
