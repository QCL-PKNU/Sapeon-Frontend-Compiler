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
        self._tf_graph = None
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

        # second, construct aixh blocks that contain successive IR nodes
        logging.warning("** YOUNGSUN - Need to determine how to organize a block for hardware acceleration **")
        logging.warning("** YOUNGSUN - Need to determine how to calculate the profit of hardware acceleration **")

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
                err = self.__perform_block_eval(ir_block)
                if err is AxfcError.SUCCESS and ir_block.nodes:
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
    
    #perform block evaluation
    #eval input and output of every node in the block
    def __perform_block_eval(self, ir_block: AxfcIRBlock, eval_modes = ["INPUT", "OUTPUT"]) -> AxfcError:
        
        if not ir_block or not ir_block.nodes or not eval_modes:
            return AxfcError.INVALID_PARAMETER
        
        #evaluation block input
        if "INPUT" in eval_modes:
            self.__eval_block_input(ir_block)
        
        #evaluation block output
        if "OUTPUT" in eval_modes:
            self.__eval_block_output(ir_block)

            #set block output node
            self.__set_block_output_node(ir_block)
        
        #set block input
        ir_block.input_nodes = self.__find_block_input_node(ir_block)
            
        return AxfcError.SUCCESS
    
    def __set_block_output_node(self, ir_block: AxfcIRBlock) -> AxfcError:
        
        output_node = self.__get_block_output_node(ir_block)
        
        if output_node:
            output_node.is_output = True
            ir_block.output_node = output_node
        
        return AxfcError.SUCCESS
    
    #For evaluating the block input and every node input
    def __eval_block_input(self, ir_block: AxfcIRBlock) -> AxfcError:
        
        #Get block inputs from first node
        # block_inputs = ir_block.nodes[0].preds
        block_inputs = self.__find_block_input_node(ir_block)
        
        #set block input node
        # ir_block.input_nodes = block_inputs
        
        #Set eval_flag to False to prepare node for evaluation
        for node in ir_block.nodes:
            node.eval_flag = False
        
        for node in ir_block.nodes:
            #skip node that is already evaluated
            if node.eval_flag:
                continue
            #perform node input evaluation
            self.__eval_node_input(node, ir_block, block_inputs)
    
    #To get the last node of the block
    #which is block output node
    def __get_block_output_node(self, ir_block: AxfcIRBlock) -> AxfcIRNode:
        
        for node in reversed(ir_block.nodes):
            for succ_node in node.succs:
                if succ_node not in ir_block.nodes:
                    return node
    
    #To get the input node from outside block
    def __find_block_input_node(self, ir_block: AxfcIRBlock) -> list:
        
        for node in ir_block.nodes:
            node.eval_flag = False
        
        input_node_list = []
        for node in ir_block.nodes:
            
            err, pred_nodes = self.__find_node_input(node, ir_block)
            if err is AxfcError.SUCCESS:
                input_node_list += pred_nodes
        
        return input_node_list
    
    #To get the node node input or preds if it is in the block
    #then it is a valid node for block but if it is outside of the block
    #then it is not a valid node.
    def __find_node_input(self, ir_node:AxfcIRNode, ir_block: AxfcIRBlock):
        
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.INVALID_PARAMETER, [ir_node]
        
        #check if pred_node is in ir_block
        for pred_node in ir_node.preds:
            #skip Const
            if pred_node.op == "Const":
                continue
            
            #if pred_node is in ir_block then its preds can not be the block input
            if pred_node in ir_block.nodes:
                return AxfcError.INVALID_PARAMETER, [ir_node]
        
        #ir_node does not contains preds in ir_block
        #then its preds are the input of the block
        return AxfcError.SUCCESS, ir_node.preds
            
    
    #if node in block connnect to outside node (except const)
    #put the node as invalid, if it's const, clone const add to aix graph
    def __eval_node_input(self, ir_node: AxfcIRNode, ir_block: AxfcIRNode, block_inputs: list) -> AxfcError:
        
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.SUCCESS

        #Check if pred node is in block_inputs
        for pred_node in ir_node.preds:
            if pred_node in block_inputs:
                return AxfcError.SUCCESS

        #check if pred node not in ir_block
        for pred_node in ir_node.preds:
            #if pred_node not in ir_block also is not a Const and Pad Node
            #Set it to not supported and remove from ir_block
            if pred_node not in ir_block.nodes and pred_node.op not in ["Const", "Pad", "Identity"]:
                ir_node.is_aixh_support = False
                ir_block.nodes.remove(ir_node)
        
        return AxfcError.SUCCESS
        
    
    #For evaluating block output to make a valid ir block 
    #A valid ir block can contain only one output node to outside of the block
    def __eval_block_output(self, ir_block: AxfcIRBlock, output_node = None) -> AxfcError: 
        
        if not ir_block or not ir_block.nodes:
            return AxfcError.INVALID_PARAMETER
        
        #Set eval_flag to False
        for node in ir_block.nodes:
            node.eval_flag = False
        
        #eval output with the last node
        # self.__eval_block_output(ir_block.nodes[0], ir_block, ir_block.nodes[-1])
        
        #---------------------------------
        for node in reversed(ir_block.nodes):
            if node.op != "Const":
                output_node = node
                break
        #--------------------------
        # if not output_node:
            
        #     # get last node of the block
        #     for node in reversed(ir_block.nodes):
        #         if node.op != "Const":
        #             output_node = node
        #             break
        #----------------------------
        
        for node in ir_block.nodes:
            #skip node that is already evaluated
            if node.eval_flag:
                continue
            
            err, expected_output_node = self.__eval_node_output(node, ir_block, output_node)
            if err is AxfcError.INVALID_NODE:
                break
            
        if err is not AxfcError.SUCCESS:
            self.__eval_block_output(ir_block, expected_output_node)
        
        return AxfcError.SUCCESS
    
    #For check the ir_node's succs if the succs is outside of the block
    #then set it as the expected output until there is only one node 
    #connecting to outside of the block 
    def __eval_node_output(self, ir_node:AxfcIRNode, ir_block:AxfcIRBlock, output_node:AxfcIRNode):
        
        if not ir_node.eval_flag:
            ir_node.eval_flag = True
        else:
            return AxfcError.SUCCESS, ir_node

        #Skip Const
        if ir_node.op == "Const":
            return AxfcError.SUCCESS, ir_node
        
        for succ_node in ir_node.succs:
            if succ_node not in ir_block.nodes and ir_node != output_node:
                #remove output_node because it is invalid 
                #as it is not the output of the node
                # ir_block.nodes.remove(output_node)
                self.__remove_node_succ_from_block(output_node, ir_block)
                
                #return the current ir_node as output_node
                return AxfcError.INVALID_NODE, ir_node
        
        return AxfcError.SUCCESS, ir_node

    #For removing node and its succs from the block
    def __remove_node_succ_from_block(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> AxfcError:
        
        if ir_node not in ir_block.nodes:
            return AxfcError.SUCCESS
        else:
            ir_block.nodes.remove(ir_node)
            # ir_node.eval_flag = False
        
        for succ_node in ir_node.succs:
            self.__remove_node_succ_from_block(succ_node, ir_block)
        
        return AxfcError.SUCCESS
    
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
        for succ in ir_node.succs:
            if not succ.is_aixh_support:
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
