import logging
import os

import torch
import torch.fx

from torchvision import models
from torch.autograd import Variable

from AxfcError          import AxfcError
from AxfcIRBuilder     import AxfcIRBuilder
from AxfcIRNode         import AxfcIRNode

import util.AxfcUtil    as _util

#######################################################################
# AxfcPTBuilder class
#######################################################################
class AxfcPTBuilder(AxfcIRBuilder):

    #The constructure
    def __init__(self, md):
        super().__init__(md)
        self.__pt_graph = None
        self.model_input = None

        #to store ir readout data
        self._ir_symtab_cnt = dict()
    
    #This function is used to read out the PyTorch mode;
    #You may use PyTorch generic library to perform it.
    def _read_model_graph(self, model_path: str, state_path: str) -> AxfcError:
        
        #write log
        logging.info("AxfcPyTorchBuilder:read_model_graph - path: %s", model_path)

        #path for model and model_state
        # model_path = state_path = ""

        # #find each model, model_state file
        # for file in os.listdir(path):
        #     if file.endswith(".pt"):
        #         model_path = os.path.join(path, file)
        #     elif file.endswith(".pth"):
        #         state_path = os.path.join(path, file)
        #     else:
        #         return AxfcError.INVALID_FILE_PATH

        #load pytorch model
        pt_model: torch.nn.Module = torch.load(model_path)
        
        #update state_dict
        pt_model.load_state_dict(torch.load(state_path))

        #set model in evaluation mode
        pt_model.eval()

        #generate graph_module by applying symblolic trace
        graph_module = torch.fx.symbolic_trace(pt_model)

        #create input placeholders for the graph
        input_placeholder = Variable(torch.randn(1, 3, 244, 244))

        #extract graph from graph_module
        self.__pt_graph = graph_module.graph

        return AxfcError.SUCCESS
    

    #This function is used to map the readout ir to AIX IR node
    #as well as building the computation AIX IR graph
    def _build_naive_ir(self, model_path: str, state_path: str) -> AxfcError:

        #read pytorch graph
        err = self._read_model_graph(model_path, state_path)

        if err is not AxfcError.SUCCESS:
            return err
        
        #translate from PyTorch graph to AIXIR
        pt_graph_def: torch.fx.graph = self.__pt_graph

        if pt_graph_def is None:
            #sholud respond INVALID_PT_GRAPH
            return AxfcError.INVALID_IR_GRAPH
        
        #build ir node for placeholders
        for pt_node_def in pt_graph_def.nodes:
            if pt_node_def.op is "placeholder":
                #append placeholder nodes into ir_sym
                err = self.__append_node_sym_ir(pt_node_def, op = "placeholder")
            elif pt_node_def.op is "output":
                #append output node into ir_sym
                err = self.__append_node_sym_ir(pt_node_def, op = "output")
            else:
                #append node into ir_sym
                err = self.__append_node_sym_ir(pt_node_def, op = pt_node_def.target)

                err = self.__append_node_def(pt_node_def)

            if err is not AxfcError.SUCCESS:
                return err
            
        #connect node pred/succ
        for pt_node_def in pt_graph_def.nodes:
            err = self.__connect_node_def(pt_node_def)
            if err is not AxfcError.SUCCESS: 
                return err

        return AxfcError.SUCCESS
    

    def __append_node_sym_ir(self, pt_node_def, op = None) -> AxfcError:

        #initializing ir node
        ir_node         = AxfcIRNode(pt_node_def)
        ir_node.name    = pt_node_def.name

        if op:
            ir_node.op = op

        #ops
        self._ir_symtab[pt_node_def.name] = ir_node

        return AxfcError.SUCCESS
    
    def __append_node_def(self, pt_node_def) -> AxfcError:

        #get ir node
        ir_node = self._ir_symtab.get(pt_node_def.name)

        #set ir node operation
        if ir_node.op is None:
            ir_node.op = pt_node_def.target

        #check the node is supported by AIXH hardware
        layer_info = self._md.get_layer_info(ir_node.op)

        if self._md.get_aixh_support(ir_node.op) and not self._md.BREAK_POINT_CONDITION:
            ir_node.is_aixh_support = True
            ir_node.aixh_profit = layer_info.profit
        else:
            ir_node.is_aixh_support = False
            ir_node.aixh_profit = 0

        return AxfcError.SUCCESS
    
    def __connect_node_def(self, pt_node_def) -> AxfcError:

        #get ir node
        ir_node = self._ir_symtab.get(pt_node_def.name)

        if not ir_node:
            logging.error("AxfcONNXIRBuilder:_connect_node_def ir_node: %s not found", pt_node_def.name)

        #Const and input are not required to connect
        if ir_node.op != None and ir_node.op not in ["placeholder, output"]:
            for pred_name in pt_node_def.all_input_nodes:
                #get ir node
                pred_node = self._ir_symtab.get(pred_name)
                if not pred_node:
                    logging.error("AxfcONNXIRBuilder:_connect_node_def ir_node: %s pred not found", pt_node_def.name)
                    return AxfcError.PRED_NODE_NOT_FOUND
                
                pred_node.succs.append(ir_node)
                ir_node.preds.append(pred_node)

        #add node to ir graph
        self._ir_graph.append_node(ir_node)
                    
        return AxfcError.SUCCESS