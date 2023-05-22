import logging

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
    def _read_model_graph(self, path: str) -> AxfcError:
        
        #write log
        logging.info("AxfcPyTorchBuilder:read_model_graph - path: %s", path)

        #load pytorch model
        pt_model = torch.load(path)
        
        #update state_dict
        pt_model.load_state_dict(torch.load("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/resnet50_state.pth"))

        pt_model.eval()

        graph_module = torch.fx.symbolic_trace(pt_model)

        # # Create input placeholders for the graph
        input_placeholder = Variable(torch.randn(1, 3, 244, 244))

        #torch graph
        pt_graph = graph_module.graph

        for node in pt_graph.nodes:
            # Access the operation of the node
            op = node.op
            print("Operation:", op)

            # Access the inputs of the node
            inputs = node.all_input_nodes
            print("Inputs:", inputs)

            # # Access the outputs of the node
            # outputs = node.all_output_nodes
            # print("Outputs:", outputs)

            # Access the target of the node (if applicable)
            target = node.target
            print("Target:", target)

            # Access any additional attributes or metadata of the node
            if node.op != "placeholder":
                attributes = node.attrs
                print("Attributes:", attributes)

            print("---------------------------------")


        return AxfcError.SUCCESS
        # return NotImplementedError()
    

    #This function is used to map the readout ir to AIX IR node
    #as well as building the computation AIX IR graph
    def _build_naive_ir(self, path: str) -> AxfcError:
        return NotImplementedError()