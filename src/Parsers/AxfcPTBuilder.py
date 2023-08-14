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

#######################################################################
# AxfcPTBuilder class
#######################################################################
class AxfcPTBuilder(AxfcIRBuilder):

    #The constructure
    def __init__(self, md):
        super().__init__(md)
        self.__pt_model = None
        self.__pt_graph = None
        self.model_input = None


    def __find_matched_op(self, node_name) -> torch.nn:
        # operator list used in ResNet50
        op_list = ['conv', 'relu', 'bn', 'add', 'downsample', 'maxpool', 'avgpool', 'flatten', 'fc']

        # if op is in target name, return op
        # For example, op is 'conv' and target is 'layer1.0.conv1'
        for op in op_list:
            if op in node_name:
                return op
                
    
    #This function is used to read out the PyTorch mode;
    #You may use PyTorch generic library to perform it.
    def _read_model_graph(self, model_path: str) -> AxfcError:
        
        #write log
        logging.info("AxfcPyTorchBuilder:read_model_graph - path: %s", model_path)

        #load pytorch model
        pt_model: torch.nn.Module = torch.load(model_path)

        self.__pt_model = pt_model

        #create input placeholders for the graph
        input_placeholder = torch.randn(1, 3, 244, 244)
        
        pt_model(input_placeholder)

        #set model in evaluation mode
        pt_model.eval()

        #generate graph_module by applying symblolic trace
        graph_module = torch.fx.symbolic_trace(pt_model)

        #extract graph from graph_module
        self.__pt_graph = graph_module.graph

        return AxfcError.SUCCESS
    

    #This function is used to map the readout ir to AIX IR node
    #as well as building the computation AIX IR graph
    def _build_naive_ir(self, model_path: str) -> AxfcError:

        #read pytorch graph
        err = self._read_model_graph(model_path)

        if err is not AxfcError.SUCCESS:
            return err
        
        #translate from PyTorch graph to AIXIR
        pt_graph_def: torch.fx.graph = self.__pt_graph

        if pt_graph_def is None:
            #sholud respond INVALID_PT_GRAPH
            return AxfcError.INVALID_IR_GRAPH
        
        ### To make the symbolic table, split the process into two sub-process
        ## 1. Read the Inputs of node [by. def)__build_ir_symtab_inputs]
        ## 2. Read the original node definition [by. def)__build_ir_symtab_def]

        # ## First,
        # # Add the Inputs of nodes into symbolic table
        # # Ex. weight, biases, running_mean, etc..
        # for input_name, param in self.__pt_model.state_dict().items():            
        #     err = self.__append_node_sym_ir_inputs(input_name, param, op = "Const")


        ## Second,
        # Build ir node symbolic table
        for pt_node_def in pt_graph_def.nodes:
            if pt_node_def.op == "placeholder":
                # append placeholder nodes into ir_sym
                # placeholder is the input of model
                err = self.__append_node_sym_ir(pt_node_def, op = "Input")
            elif pt_node_def.op == "output":
                #append output node into ir_sym
                err = self.__append_node_sym_ir(pt_node_def, op = "Output")
            else:
                # Since, the pt_node.op is function for calling module and method and so on
                # extract the operator from the pt_node definition like 'conv', 'bn'
                node_op = self.__find_matched_op(pt_node_def.name)

                # Make the symbolic table for pt_node_def
                err = self.__append_node_sym_ir(pt_node_def, op = node_op)

                # Append info to ir_node
                err = self.__append_node_def(pt_node_def)

            if err is not AxfcError.SUCCESS:
                return err
            
        ## Connect node pred/succ
        # Pred includes the input (weight, biase, etc) and previous node
        for pt_node_def in pt_graph_def.nodes:
            err = self.__connect_node_def(pt_node_def)
            if err is not AxfcError.SUCCESS: 
                return err

        return AxfcError.SUCCESS
    
    # def __append_node_sym_ir_inputs(self, input_name, param, op = None) -> AxfcError:
    #     # make custom node definition
    #     input_def = dict()
    #     input_def[input_name] = param

    #     # Initialize ir node
    #     ir_node = AxfcIRNode(input_def)

    #     ir_node.name = input_name

    #     if op:
    #         ir_node.op = op

    #     self._ir_symtab[input_name] = ir_node

    #     return AxfcError.SUCCESS

    ## This method is used to make the symbolic table for the pt node definition
    def __append_node_sym_ir(self, pt_node_def, op = None) -> AxfcError:

        #initializing ir node
        ir_node         = AxfcIRNode(pt_node_def)

        # NOTE: node.target is the name of node
        # such as 'layer1.0.conv1'
        ir_node.name    = pt_node_def.name     

        if op:
            ir_node.op = op

        # Make symbolic table
        # table consists of {'node_name': node_definition}
        self._ir_symtab[pt_node_def.name] = ir_node

        return AxfcError.SUCCESS
    
    ## This method is used to append the info of pt node definition to ir_node
    def __append_node_def(self, pt_node_def) -> AxfcError:

        # Get ir node
        ir_node = self._ir_symtab.get(pt_node_def.name)

        # Set ir node operation
        if ir_node.op is None:
            ir_node.op = self.__find_matched_op(pt_node_def.name)

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
    
    ## This function is used to connect all nodes each other
    def __connect_node_def(self, pt_node_def) -> AxfcError:

        #get ir node
        ir_node = self._ir_symtab.get(pt_node_def.name)

        if not ir_node:
            logging.error("AxfcPyTorchIRBuilder:_connect_node_def ir_node: %s not found", pt_node_def.name)

        #Const and input are not required to connect
        if ir_node.op != None and ir_node.op not in ["Input", "Output", "Const"]:
            for pred_name in pt_node_def.all_input_nodes:
                #get ir node
                pred_node = self._ir_symtab.get(pred_name.name)
                if not pred_node:
                    logging.error("AxfcPyTorchIRBuilder:_connect_node_def ir_node: %s pred not found", pt_node_def.name)
                    return AxfcError.PRED_NODE_NOT_FOUND
                
                # Succs denotes the next node
                pred_node.succs.append(ir_node)

                # Preds denotes the previous node
                ir_node.preds.append(pred_node)

        #add node to ir graph
        self._ir_graph.append_node(ir_node)
                    
        return AxfcError.SUCCESS