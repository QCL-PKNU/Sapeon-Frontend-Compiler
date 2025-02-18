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

from abc import ABC, abstractmethod
from typing import List, Tuple
from AxfcIRGraph import *
from AxfcMachineDesc import AxfcMachineDesc
from AxfcError import AxfcError

#######################################################################
# AxfcIRBuilder class
#######################################################################

class AxfcIRBuilder(ABC):
    """
    A builder class for creating an Intermediate Representation (IR) graph
    from a given machine description.

    Attributes:
        _md: A reference to the AIX machine description.
        _ir_graph: The resulting AIXIR graph built from the input.
        _ir_symtab: A symbol table for the IR graph.
    """


    def __init__(self, md):
        self._md = md
        self._ir_graph = None
        self._ir_symtab = None


    def build_ir(self, model_path: str) -> {AxfcError, AxfcIRGraph}:
        self._ir_symtab = {}
        self._ir_graph = AxfcIRGraph(self._ir_graph)

        # Generates naive IR from dl model
        err = self._build_naive_ir(model_path)
        if err is not AxfcError.SUCCESS:
            logging.warning(f"AxfcIRBuilder: Failed to generate a naive IR with error: {err}")
            return err, None
        
        # Finds AIXH blocks, then convert each block into AIXGraph
        err = self.__find_aixh_blocks()
        if err is not AxfcError.SUCCESS:
            logging(f"AxfcIRBuilder: Failed to find AIXH blocks with error: {err}")
            return err, None
        
        for ir_block in self._ir_graph.blocks:
            if not ir_block.is_aixh_support:
                continue

            # Analyze local liveness of the AIXH blocks
            err = ir_block.analyse_liveness()
            if err is not AxfcError.SUCCESS:
                logging.warning(f"AxfcIRBuilder: Failed to analyze local liveness of the block ({ir_block.id}) with error: {err}")
                return err, None

            # just for debugging - YOUNGSUN
            # if ir_block.is_aixh_support:
            #    print(ir_block)

        return AxfcError.SUCCESS, self._ir_graph


    def __find_aixh_blocks(self) -> AxfcError:
        """
        This method is used to find AIXH blocks comprised of AIXH-supported nodes. We employ a maximal munching 
        scheme to find the longest successive AIXH-supported nodes and build up a block with the nodes.

        Returns:
            AxfcError: An error code indicating the success of the block evaluation.
        """
        logging.info("AxfcIRBuilder:find_aixh_blocks")

        if self._ir_graph is None:
            logging.warning("IR graph is invalid or not initialized.")
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

            # When the first ir_node pass the perform_block_eval, find_block_input_node,
            # eval_flag of other nodes is set into True
            # Therefore, ignore if ir_node is already evaluated
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
    
    
    def __perform_block_eval(self, ir_block: AxfcIRBlock) -> AxfcError:
        """
        Performs block evaluation by identifying and removing nodes that fall under block constraints,
        affecting the transformation of the block into an AIX graph.

        Args:
            ir_block (AxfcIRBlock): The IR block to be evaluated.

        Returns:
            AxfcError: An error code indicating the success of the block evaluation.
        """

        # Find the input nodes of the block
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

    
    def __remove_succ_from_block(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock):
        """Recursively removes a node and its successors from the specified block if they belong to that block.

        Args:
            ir_node (AxfcIRNode): The node to start removal from.
            ir_block (AxfcIRBlock): The block from which nodes will be removed.
            visited (set): Set of already visited nodes to prevent infinite recursion in cyclic graphs.
        """
        
        # Remove node if it exists in block
        if ir_node in ir_block.nodes:
            ir_node.eval_flag = False # For reevaluation
            ir_block.nodes.remove(ir_node)
        
        for succ_node in ir_node.succs:
            if succ_node in ir_block.nodes:
                self.__remove_succ_from_block(succ_node, ir_block)


    def __get_block_output_node_list(self, ir_block: AxfcIRBlock, set_inout = False):
        """Retrieves a list of output nodes from the specified IR block. An output node is defined as a node
        whose successor is outside the block or requires the node as an input.

        Args:
            ir_block (AxfcIRBlock): The IR block from which to retrieve output nodes.
            set_output (bool): If True, marks each node identified as an output node. Defaults to False.

        Returns:
            List[AxfcIRNode]: A list of nodes that are considered outputs of the block.
        """
        output_node_list = []

        for node in reversed(ir_block.nodes):
            if len(node.succ) == 0:
                node.is_output = True

            for succ_node in node.succs:
                if succ_node not in ir_block.nodes and node.op not in ["Const", "Identity", "Pad"]:
                    if node not in output_node_list:
                        output_node_list.append(node)

                        if set_inout:
                            node.is_output = True

                    # break
        
        return output_node_list
    
    
    def __find_block_input_node(self, ir_block: AxfcIRBlock, set_inout = False) -> list:
        """
        Identifies the input nodes of the given block. An input node is defined as a node
        within the block that has at least one predecessor outside of the block.

        Args:
            ir_block (AxfcIRBlock): The block to analyze.
            mark_as_input (bool): If True, marks identified nodes as input nodes.

        Returns:
            list: A list of AxfcIRNode instances representing the block's input nodes.
        """

        # **Leanghok - using eval_flag here might cause error on finding block
        # Temporary set evaluation flag as False
        for node in ir_block.nodes:
            node.eval_flag = False
        
        input_node_list = []
        for node in ir_block.nodes:
            
            # Find the input node of current ir_node to check whether it is out/inside from block
            err, pred_nodes = self.__find_node_input(node, ir_block)
            if err is AxfcError.SUCCESS:
                input_node_list += pred_nodes

                if set_inout and "FusedBatchNorm" not in node.name and len(pred_nodes) > 0:
                    node.is_input = True
        
        return input_node_list
    

    def __find_node_input(self, ir_node:AxfcIRNode, ir_block: AxfcIRBlock):
        """
        Identifies the predecessors of a given node to determine if they are
        inside or outside of the specified block. Preds outside the block
        are considered inputs to the block.

        Args:
            ir_node (AxfcIRNode): The node whose predecessors are to be checked.
            ir_block (AxfcIRBlock): The block against which the predecessors are checked.

        Returns:
            (AxfcError, list of AxfcIRNode): Tuple containing an error code and a list
            of nodes that are predecessors of `ir_node` and outside of `ir_block`.
        """
        
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

            # from collections.abc import Iterable
            # if isinstance(pred_node, Iterable) is False:
            #     pred_node_inputs.append(pred_node)
            #     continue
            
            #if pred_node is in ir_block then its preds can not be the block input
            if pred_node not in ir_block.nodes and "Pad" not in pred_node.name:
                # return AxfcError.INVALID_PARAMETER, [ir_node]
                pred_node_inputs.append(pred_node)
        
        #ir_node does not contains preds in ir_block
        #then its preds are the input of the block
        return AxfcError.SUCCESS, pred_node_inputs
            
    
    def __perform_maximal_munch(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> AxfcError:
        """
        Performs maximal munch algorithm to recursively find and group the longest sequence of 
        successive AIXH-supported nodes starting from a given node.

        Args:
            ir_node (AxfcIRNode): The start node to perform maximal munching from.
            ir_block (AxfcIRBlock): The IR block to which successive AIXH-supported nodes are added.

        Returns:
            AxfcError: Status of the maximal munching operation.
        """
        logging.info("AxfcIRBuilder:perform_maximal_munch")

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
            ir_block.nodes.append(ir_node) # append ir_node into nodes of ir_block
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
    
    
    @abstractmethod
    def _read_model_graph(self, path: str):
        """Read a grpah from input file of given path."""
        return NotImplementedError()
    

    @abstractmethod
    def _build_naive_ir(self, path: str):
        """Constructs a naive Intermediate Representation (IR) using graph from AI model."""
        return NotImplementedError()
    

    ## For debugging
    def __str__(self):
        pass