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

from typing import List, Tuple
from AxfcIRGraph import *
from AxfcMachineDesc import *
from AxfcError import AxfcError


#######################################################################
# AxfcIRBuilder class
#######################################################################

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
        #remove _tf_graph from AxfcIRBuilder
        #self._tf_graph = None 
        self._ir_graph = None
        self._ir_symtab = None

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

        # create a new symbol table and IR graph
        self._ir_symtab = dict()
        self._ir_graph = AxfcIRGraph(self._ir_symtab)

        # build a naive IR using the IR builder of a specific type
        err = self._build_naive_ir(path)
        if err is not AxfcError.SUCCESS:
            logging.warning("build naive IR error: %s", err)
            return err, None

        # find AIXH blocks to be translated into AIXGraphs
        err = self.__find_aixh_blocks()
        if err is not AxfcError.SUCCESS:
            logging.warning("find AIXH blocks: %s", err)
            return err, None

        # for all the blocks of the IR graph
        for ir_block in self._ir_graph.blocks:
            # ignore blocks not supported by hardware
            if not ir_block.is_aixh_support:
                continue

            # perform the local liveness analysis for all the AIXH blocks
            # to resolve the input and output of them
            err = ir_block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                logging.warning("analyse liveness: block %d", ir_block.id)
                return err, None

            # just for debugging - YOUNGSUN
            # if ir_block.is_aixh_support:
            #    print(ir_block)

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

        # **Leanghok - The warning sections has been implemented
        # second, construct aixh blocks that contain successive IR nodes
        # logging.warning("** YOUNGSUN - Need to determine how to organize a block for hardware acceleration **")
        # logging.warning("** YOUNGSUN - Need to determine how to calculate the profit of hardware acceleration **")

        # filter nodes that supported by AIXH
        supported_nodes = list(filter(lambda node: node.is_aixh_support, self._ir_graph.nodes))

        # indexing all ir_node
        for index, node in enumerate(supported_nodes):
            node.temp_id = index

        for ir_node in supported_nodes:

            # ignore nodes that are already evaluated
            if ir_node.eval_flag:
                continue

            # create a new IR block and perform maximal munching
            ir_block = AxfcIRBlock()

            err = self.__perform_maximal_munch(ir_node, ir_block)
            if err is AxfcError.SUCCESS:
                
                #Perform block node evaluation for aixh support block
                # err = self.__perform_block_eval(ir_block)
                # if err is AxfcError.SUCCESS and ir_block.nodes:
                #     self._ir_graph.append_block(ir_block)
                
                #-----------
                err = self.__perform_block_eval(ir_block)

                if err == AxfcError.SUCCESS:
                    #set block input
                    ir_block.input_nodes = self.__find_block_input_node(ir_block, set_inout = True)
                    #set block output
                    ir_block.output_nodes = self.__get_block_output_node_list(ir_block, set_inout= True)
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

        # sort the nodes in each block by layer id
        for block in self._ir_graph.blocks:
            block.nodes.sort(key=lambda node: node.layer_id)

        return AxfcError.SUCCESS
    
    
    ## This method is used to perform the block evaluation by by checking all nodes
    #  and remove any node that fall under the block constraints and effect 
    #  the transformation of the block to aix graph
    #
    #  @param self this object
    #  @param ir_block AxfcIRBlock type
    #  @return error info
    def __perform_block_eval(self, ir_block: AxfcIRBlock) -> AxfcError:
        

        input_nodes = self.__find_block_input_node(ir_block)

        node_to_remove = []
        for node in input_nodes:
            for pred_node in node.preds:
                if pred_node in ir_block.nodes:
                    node_to_remove.append(node)
                    break
        
        for node in node_to_remove:
            self.__remove_succ_from_block(node, ir_block)
        
        return AxfcError.SUCCESS

    ## This method is used to perform successors removing of a specific node
    #  that successor must be a node of the block 
    #
    #  @param self this object
    #  @param ir_node AxfcIRNode type
    #  @param ir_block AxfcIRBlock type
    def __remove_succ_from_block(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock):
        
        #remove node if it exists in block
        if ir_node in ir_block.nodes:
            #Set eval flag to fail so can revaluate for another block
            ir_node.eval_flag = False
            ir_block.nodes.remove(ir_node)
        
        for succ_node in ir_node.succs:
            if succ_node in ir_block.nodes:
                self.__remove_succ_from_block(succ_node, ir_block)

    ## This method is used to get the output nodes of the block as a list
    #  the output is defined by if the node is connected to another node outside 
    #  then it is define as the output of the block since the another node outside 
    #  of the block requires that node as an input
    #
    #  @param self this object
    #  @param ir_block AxfcIRBlock type
    #  @param set_input Boolean type is used as conditonal operation whether or not to 
    #  set the node as an output node
    #  @return list of AxfcIRNode which refers block's output nodes
    def __get_block_output_node_list(self, ir_block: AxfcIRBlock, set_inout = False):

        output_node_list = []

        for node in reversed(ir_block.nodes):
            for succ_node in node.succs:
                if succ_node not in ir_block.nodes and node.op not in ["Const", "Identity", "Pad"]:
                    if node not in output_node_list:
                        output_node_list.append(node)

                        if set_inout:
                            node.is_output = True

                    # break
        
        return output_node_list
    
    ## This method is used to get the input nodes of the the block as a list
    #  the input node is defined by if the node in the block has any predecessor
    #  outside of the block.
    #
    #  @param self this object
    #  @param ir_block AxfcIRBlock type
    #  @param set_input Boolean type is used as conditonal operation whether or not to 
    #  set the node as an input node
    #  @return list of AxfcIRNode which refers block's input nodes
    def __find_block_input_node(self, ir_block: AxfcIRBlock, set_inout = False) -> list:

        # **Leanghok - using eval_flag here might cause error on finding block
        for node in ir_block.nodes:
            node.eval_flag = False
        
        input_node_list = []
        for node in ir_block.nodes:
            
            err, pred_nodes = self.__find_node_input(node, ir_block)
            if err is AxfcError.SUCCESS:
                input_node_list += pred_nodes

                if set_inout and "FusedBatchNorm" not in node.name and len(pred_nodes) > 0:
                    node.is_input = True
        
        return input_node_list
    
    ## This method is used to identify the node inputs/preds as if 
    #  it is in the block or outside of the block
    #
    #  @param self this object
    #  @param ir_node AxfcIRNode type
    #  @param ir_block AxfcIRBlock type
    #  @return error info, list of AxfcIRNode which refers node's input
    def __find_node_input(self, ir_node:AxfcIRNode, ir_block: AxfcIRBlock):
        
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.INVALID_PARAMETER, [ir_node]

        pred_node_inputs = []
        #check if pred_node is in ir_block
        # **Leanghok - block cannot find block's input node when input is actually constant
        # conflict from having others constant ignore
        for pred_node in ir_node.preds:
            #skip Const
            if pred_node.op == "Const":
                continue
            
            #if pred_node is in ir_block then its preds can not be the block input
            if pred_node not in ir_block.nodes and "Pad" not in pred_node.name:
                # return AxfcError.INVALID_PARAMETER, [ir_node]
                pred_node_inputs.append(pred_node)
        
        #ir_node does not contains preds in ir_block
        #then its preds are the input of the block
        return AxfcError.SUCCESS, pred_node_inputs
            
    
    ## This method performs maximal munch algorithm to
    #  recursively find the longest successive AIXH-supported nodes.
    #
    # @param self this object
    # @param ir_node a start node to perform maximal munching
    # @param an IR block of the successive IR nodes supported by the AIX hardware
    # @return error info
    def __perform_maximal_munch(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> AxfcError:
        # logging.info("AxfcIRBuilder:perform_maximal_munch")

        if ir_node is None or ir_block is None:
            return AxfcError.INVALID_PARAMETER

        # skip if this node has already been evaluated
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.SUCCESS
        
        # skip if node is a Const
        if ir_node.op == "Const":
            return AxfcError.SUCCESS

        # skip if this node is not supported by hardware
        if ir_node.is_aixh_support:
            ir_node.layer_id = ir_node.temp_id
            ir_node.block_ref = ir_block
            ir_block.nodes.append(ir_node)
        else:
            return AxfcError.SUCCESS
        
        #end block if succ is not supported by hardware
        # for succ in ir_node.succs:
        #     if not succ.is_aixh_support:
        #         return AxfcError.SUCCESS

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

    #######################################################################
    ## Abstract methods
    #######################################################################

    ## This method is used to read a tensorflow graph from an input file in the given path.
    #
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _read_model_graph(self, path: str):
        return NotImplementedError()

    ## This method is used to construct a naive AIXIR using a tensorflow graph.
    #
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _build_naive_ir(self, path: str):
        return NotImplementedError()
