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
#   [Before:High Performance Computing Laboratory (hpcl.pknu.ac.kr)]
#######################################################################

import torch
import torch.nn as nn
import torch.fx as fx
from AxfcIRGraph import *
from AxfcMachineDesc import *
import torchvision.models as models 


class AIXOpLayer(nn.Module):

    # @var path
    # path of aix graph

    # @var outputs
    # output tensor

    ## The constructor
    # def __init(self, path: str, outputs):
    #     self.attrs = {"aix_graph_path": path}
    #     self.outputs = outputs

    ## The constructor
    def __init__(self, path: str, outputs):
        super(AIXOpLayer, self).__init__()
        self.attrs = {"aix_graph_path": path}
        self.name = "AixOp"
        self.op = "AixOp"
        self.inputs = None
        self.outputs = outputs

    ## Forward func is used to define the actual behaviour of custom op.
    #
    # @param self this object
    # @param x output tensor of predecessor node
    def forward(self, x):
        self.inputs = [x]
        scalar = (x * 0 + 1).mean()
        return self.outputs * scalar

## TODO: This method is used to define symbolic representation for AixOp, still required extending onnx runtime
#
# @param self this object
def symbolic_aixop(g, input, *args, **kwargs):
    return g.op("AixOp", input, args, kwargs)

torch.onnx.register_custom_op_symbolic('::AixOp', symbolic_aixop, opset_version=11)

class AxfcPTWriter:

    ## @var __frozen_model_path
    # the path to the pre-trained and 'frozen' model

    # @var __aix_graph_path
    # the path of aix graph
    
    ## @var __ir_graph
    # an AIX graph that will be used for writing the laucher 

    # @var hook_outputs
    # to store outputs from the registered hooks during the forward pass

    ## The constructor
    # def __init(self, ir_graph: str, frozen_model_path: str, aix_graph_path: str):
    #     self.frozen_model_path = frozen_model_path
    #     self.aix_graph_path = aix_graph_path
    #     self.__ir_graph = ir_graph

    ## The constructor
    def __init__(self,
                 frozen_model_path: str,
                 aix_graph_path: str,
                ir_graph: AxfcIRGraph,
                md: AxfcMachineDesc):
        self.frozen_model_path = frozen_model_path
        self.aix_graph_path = aix_graph_path
        self.ir_graph = ir_graph
        self.model = torch.load(frozen_model_path)
        self.gm = fx.GraphModule = fx.symbolic_trace(self.model)
        self.hook_outputs = {}
    
    ## This method is used to extract output tensor during forward pass
    #
    # @param self this object
    # @param module nn.module that has been register hook
    def hook(self, module, input, output):
        module_key = module.__class__.__name__
        self.hook_outputs[module_key] = output.detach()

    ## This method is used to modify model by replace custom AixOp to aixh block
    #
    # @param self this object
    def get_custom_graph(self):
        tensor = self.generate_input_tensor()
        
        # loop through block and replace aixh block with custom AixOpLayer
        for block in self.ir_graph.blocks:
            if not block.is_aixh_support:
                continue
            ir_block_nodes = self.traverse_block(block)
            gr_block_nodes = self.find_cor_graph_nodes(ir_block_nodes)

            # register hook for output node to extract tensor
            if gr_block_nodes[-1].name == block.output_nodes[0].name:
                submodule = getattr(self.gm, gr_block_nodes[-1].target)
                submodule.register_forward_hook(self.hook)

            self.gm.recompile()
            self.gm(tensor)
        
            # add new module to the fx graph
            self.gm.add_module('AixOp', AIXOpLayer(
                self.aix_graph_path, list(self.hook_outputs.items())[0][1]
            ))

            # find the node before new_node
            pred_node = {}
            if 'Input' in block.input_nodes[0].op:
                pred_node = list(self.gm.graph.nodes)[0]
            else:
                for node in list(self.gm.graph.nodes):
                    if not block.input_nodes[0].op == node.op:
                        continue
                    pred_node = node

            # inserting new node
            with self.gm.graph.inserting_after(pred_node):
                new_node = self.gm.graph.create_node(
                    'call_module',
                    'AixOp',
                    args=(list(self.gm.graph.nodes)[1].args[0],)
                )
            
            # remove nodes and and rewire dependency
            for node in gr_block_nodes:
                for user, _ in list(node.users.items()):
                    user.replace_input_with(node, new_node)
                self.gm.graph.erase_node(node)

        self.gm.recompile()

        # convert custom model to onnx format
        torch.onnx.export(self.gm, tensor, 
                          self.get_model_path('onnx'), 
                          opset_version=11,
                          input_names=['data'],
                          output_names=['output'])

        # save pytorch custom model
        saved_path = self.get_model_path('pt')
        torch.save(self.gm, saved_path)
        return AxfcError.SUCCESS, saved_path
    
    ## This method is used to generate input_shape of DL model
    # 
    # @param self this object
    # @param batch_size number of training
    def generate_input_tensor(self, batch_size = 1):
        first_layer = next(self.model.children())
        if isinstance(first_layer, nn.Conv2d):
            # For Conv2D, the expected input format is (N, C, H, W)
            return torch.randn(batch_size, first_layer.in_channels, 224, 224)
        if isinstance(first_layer, nn.Linear):
            # For Linear, the expected input format is (N, in_features)
            return torch.randn(batch_size, first_layer.in_features)
        else:
            raise TypeError("Unsupported Type")
    
    ## This method is used to traverse nodes within a block and return a set of nodes.
    #
    # @param self this object
    # @param block aixh block contains nodes
    def traverse_block(self, block):
        node_to_process = list(block.input_nodes)
        processed_nodes = set()
        nodes_in_block = set()

        while node_to_process:
            node = node_to_process.pop()
            if node in processed_nodes:
                continue

            processed_nodes.add(node)
            nodes_in_block.add(node)

            for succ in node.succs:
                if succ not in block.output_nodes:
                    node_to_process.append(succ)

        nodes_in_block.add(block.output_nodes[0])
        return nodes_in_block
    
    ## This method is used to find the corresponding node in the fx graph
    #
    # @param self this objects
    # @param nodes_in_block nodes in the aixh block of ir_graph
    def find_cor_graph_nodes(self, nodes_in_block):
        corresponding_nodes = []
        for node in self.gm.graph.nodes:
            for b_node in nodes_in_block:
                if node.name == b_node.name:
                    corresponding_nodes.append(node)

        return corresponding_nodes
    

    ## This method is used to create a custom path for custom model
    #
    # @param self this object
    # @param extension type of file to be save
    def get_model_path(self, extension):
        return self.frozen_model_path.rsplit('.', 1)[0] + f'_custom.{extension}'

