#######################################################################
#   AxfcIRTranslator
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#      Sanghyeon Lee (sanghyeon@pukyong.ac.kr)
#      Pov Kimsay (povkimsay@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################

import logging
import os

import torch
import torch.fx

from torchvision import models
from torch.autograd import Variable

from AxfcError          import AxfcError
from .AxfcIRBuilder     import AxfcIRBuilder
from AxfcIRNode         import AxfcIRNode

import util.AxfcUtil    as _util

@torch.fx.wrap
def torch_randn(shape):
    return torch.randn(shape)

#######################################################################
# AxfcPTBuilder class
#######################################################################

class AxfcPTIRBuilder(AxfcIRBuilder):
    """Pytorch IR building class.

    In this class, computation graph is retrieved from the Pytorch model
    and being used for building Intermedaite Representation.

    Attributes:
        _pt_model: A Pytorch model.
        _pt_graph: A pytorch graph.
    """


    def __init__(self, md):
        super().__init__(md)
        self.__pt_model = None  
        self.__pt_graph = None 


    def __find_matched_op(self, node_name) -> torch.nn:
        """Find a matched operation from a node of PyTorch graph.
        
        Args:
            node_name: The name of the node in the computational graph.
        """

        # ResNet50 operations
        op_list = [
            'conv', 'relu', 'bn', 'add', 'downsample', 
            'maxpool', 'avgpool', 'flatten', 'fc'
        ]

        # if op is in target name, return op
        # For example, op is 'conv' and target is 'layer1.0.conv1'
        for op in op_list:
            if op in node_name:
                return op


    def _read_model_graph(self, model_path: str) -> AxfcError:
        """Read the PyTorch model.
        
        Args:
            model_path: The path to Pytorch model.
        """
        logging.info("AxfcPyTorchBuilder:read_model_graph - path: %s", model_path)

        model: torch.nn.Module = torch.load(model_path)
        input_tensor = torch_randn((1, 3, 224, 224))
        graph_module = torch.fx.symbolic_trace(model, (input_tensor, ))

        # Extract graph from graph_module
        self.__pt_graph = graph_module.graph

        return AxfcError.SUCCESS
    

    def _build_naive_ir(self, model_path: str) -> AxfcError:
        """Construct a naive AIX IR from a Pytorch Graph.

        Args:
            model_path (str): A path to pytorch model.
        """
        # Read model graph
        if (err := self._read_model_graph(model_path)) != AxfcError.SUCCESS:
            return err
        
        graph_def: torch.fx.graph = self.__pt_graph
        if graph_def is None:
            return AxfcError.INVALID_IR_GRAPH
        
        # Build ir node symbolic table
        for node_def in graph_def.nodes:
            if node_def.op in ["placeholder", "get_attr"]:
                err = self.__append_node_sym_ir(node_def, op = "Input")
            elif node_def.op == "output":
                err = self.__append_node_sym_ir(node_def, op = "Output")
            else:
                # Since, the pt_node.op is function for calling module and method and so on
                # extract the operator from the pt_node definition like 'conv', 'bn'
                node_op = self.__find_matched_op(node_def.name)

                # Make the symbolic table for pt_node_def
                err = self.__append_node_sym_ir(node_def, op = node_op)

                # Append info to ir_node
                err = self.__append_node_def(node_def)

            if err is not AxfcError.SUCCESS:
                return err
            
        ## Connect node pred/succ
        # Pred includes the input (weight, biase, etc) and previous node
        for pt_node_def in graph_def.nodes:
            err = self.__connect_node_def(pt_node_def)
            if err is not AxfcError.SUCCESS: 
                return err

        return AxfcError.SUCCESS
    

    def __append_node_sym_ir(self, node_def, op = None) -> AxfcError:
        """Append IR Node into IR symbolic table
        
        Args:
            node_def: Node definition of PyTorch model.
            op: 
        """
        ir_node = AxfcIRNode(node_def)
        ir_node.name = node_def.name
        if op:
            ir_node.op = op

        self._ir_symtab[node_def.name] = ir_node

        return AxfcError.SUCCESS
    

    def __append_node_def(self, node_def) -> AxfcError:
        """Create a new IR node from the node_def and append to IR graph.

        Args:
            node_def: Node definition of PyTorch model.
        """

        ir_node = self._ir_symtab.get(node_def.name)
        if ir_node.op is None:
            ir_node.op = self.__find_matched_op(node_def.name)

        # Check the node is supported by AIXH hardware
        layer_info = self._md.get_layer_info(ir_node.op)

        # Set the profit property
        if self._md.get_aixh_support(str(ir_node.op)) and not self._md.BREAK_POINT_CONDITION:
            ir_node.is_aixh_support = True
            ir_node.aixh_profit = layer_info.profit
        else:
            ir_node.is_aixh_support = False
            ir_node.aixh_profit = 0

        return AxfcError.SUCCESS
    
    
    def __connect_node_def(self, node_def) -> AxfcError:
        """
        Connects an IR node to its predecessors and successors. Nodes of types 'Const', 'Input',
        and 'Output' do not require connections.

        Args:
            node_def: The definition of the node from the PyTorch model.
        Returns:
            AxfcError: Error code indicating the success or failure of the operation.
        """

        # Attempt to retrieve the corresponding IR node
        ir_node = self._ir_symtab.get(node_def.name)
        if ir_node is None:
            logging.error("AxfcPyTorchIRBuilder:_connect_node_def ir_node: %s not found", node_def.name)            

        # Connect the current node with its predecessors and successors
        # Skip connection logic for 'Const', 'Input', and 'Output' nodes
        if ir_node.op != None and ir_node.op not in ["Input", "Output", "Const"]:
            for pred_name in node_def.all_input_nodes:
                # Get IR node
                pred_node = self._ir_symtab.get(pred_name.name)
                if not pred_node:
                    logging.error("AxfcPyTorchIRBuilder:_connect_node_def ir_node: %s pred not found", node_def.name)
                    return AxfcError.PRED_NODE_NOT_FOUND
                
                # Establish connections
                pred_node.succs.append(ir_node)
                ir_node.preds.append(pred_node)

        # Successfully add the node to the IR graph
        self._ir_graph.append_node(ir_node)
                    
        return AxfcError.SUCCESS