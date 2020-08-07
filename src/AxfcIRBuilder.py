#######################################################################
#   AxfcIRBuilder
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
from AxfcIRGraph import *
from AxfcMachineDesc import *

##
# AxfcIRBuilder
#
class AxfcIRBuilder:

    ## @var _md
    # AIX machine description

    ## @var _tf_graph
    # input Tensorflow graph

    ## @var _ir_graph
    # output AIXIR graph

    ## @var _ir_symtab
    # symbol table for IR graph

    ## The constructor
    def __init__(self, md):
        self._md = md

        self._tf_graph = None
        self._ir_graph = None

        self._ir_symtab = dict()

    def _read_model_graph(self, path: str):
        return NotImplementedError()

    def _build_naive_ir(self, path: str):
        return NotImplementedError()

    def _visualize_ir_graph(self):
        return NotImplementedError()

    ## This method is used to build AXI IR.
    #  1) it builds a naive IR using the given input model.
    #  2) it checks the IR nodes to be executed in hardware-manner.
    #  3) it finds AIXH IR blocks. each block consist of several AIXH IR nodes.
    #  4) it performs the liveness analysis for resolving the input and output of the blocks.
    #
    #  @param self this object
    #  @param path input path of a frozen model
    #  @return error info and an AxfcIRGraph object
    def build_ir(self, path: str) -> {AxfcError, AxfcIRGraph}:
        logging.info("AxfcIRBuilder:build_ir - path: %s", path)

        # build a naive IR using the IR builder of a specific type
        err = self._build_naive_ir(path)
        if err is not AxfcError.SUCCESS:
            logging.warning("build naive IR error: %s", err)
            return err, None

        # just for debugging - YOUNGSUN
        #self._visualize_graph()

        # find AIXH blocks to be translated into AIXGraphs
        err = self.__find_aixh_blocks()
        if err is not AxfcError.SUCCESS:
            logging.warning("find AIXH blocks: %s", err)
            return err, None

        # perform the local liveness analysis for all the AIXH blocks
        # to resolve the input and output of them
        for ir_block in self._ir_graph.blocks:
            err = ir_block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                logging.warning("analyse liveness: block %d", ir_block.id)
                return err, None

            # just for debugging - YOUNGSUN
            if ir_block.is_aixh_support:
                print(ir_block)

        return AxfcError.SUCCESS, self._ir_graph

    ## This method is used to find AIXH blocks comprised of AIXH-supported nodes.
    #  We employ a maximal munching scheme to find the longest successive AIXH-supported nodes and
    #  build up a block with the nodes.
    #
    #  @param self this object
    #  @return error info
    def __find_aixh_blocks(self) -> AxfcError:
        logging.info("AxfcIRBuilder:find_aixh_blocks")

        if self._ir_graph is None:
            return AxfcError.INVALID_IR_GRAPH

        # second, construct aixh blocks that contain successive IR nodes
        logging.warning("** YOUNGSUN - Need to determine how to organize a block for hardware acceleration **")
        logging.warning("** YOUNGSUN - Need to determine how to calculate the profit of hardware acceleration **")

        for ir_node in self._ir_graph.nodes:
            # ignore nodes that are already evaluated and not supported by AIXH
            if ir_node.eval_flag or not ir_node.is_aixh_support:
                continue

            # create a new IR block and perform maximal munching
            ir_block = AxfcIRBlock()

            err = self.__perform_maximal_munch(ir_node, ir_block)
            if err is AxfcError.SUCCESS:
                self._ir_graph.append_block(ir_block)
            else:
                return err

            # analyze the profit of accelerating the block and
            # determine whether the block will be execute in hardware-manner or not
            err = ir_block.analyze_profit()
            if err is not AxfcError.SUCCESS:
                return err

            profit_threshold = self._md.get_profit_threshold()

            if ir_block.aixh_profit >= profit_threshold:
                ir_block.is_aixh_support = True
            else:
                ir_block.is_aixh_support = False

        return AxfcError.SUCCESS

    ## This method performs maximal munch algorithm to
    #  recursively find the longest successive AIXH-supported nodes.
    #
    # @param self this object
    # @param ir_node a start node to perform maximal munching
    # @param an IR block of the successive IR nodes supported by the AIX hardware
    # @return error info
    def __perform_maximal_munch(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> AxfcError:
        #logging.info("AxfcIRBuilder:perform_maximal_munch")

        if ir_node is None or ir_block is None:
            return AxfcError.INVALID_PARAMETER

        # skip if this node has already been evaluated
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.SUCCESS

        # skip if this node is not supported by hardware
        if ir_node.is_aixh_support:
            ir_block.nodes.append(ir_node)
            ir_node.block_ref = ir_block
        else:
            return AxfcError.SUCCESS

        # perform maximal munching to the predecessors
        for pred in ir_node.preds:
            self.__perform_maximal_munch(pred, ir_block)

        # perform maximal munching to the successors
        for succ in ir_node.succs:
            self.__perform_maximal_munch(succ, ir_block)

        return AxfcError.SUCCESS

    ## For debugging
    def __str__(self):
        pass