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
#   Quantum Computing Laboratory (quantum.pknu.ac.kr)
#######################################################################

import torch
import torch.nn as nn
import torch.fx as fx
from AxfcError import AxfcError
from AxfcIRGraph import AxfcIRGraph
from AxfcMachineDesc import AxfcMachineDesc


class AIXOpLayer(nn.Module):
    """
    A custom PyTorch layer represent an AIX operation.

    Attributes:
        attrs (dict): Attributes containing the AIX graph path.
        name (str): Name of the operation.
        op (str): Type of operation, used for identification.
        inputs (list or None): Inputs to the operation.
        outputs (torch.Tensor): The predefined output 
    """


    def __init__(self, path: str, outputs):
        super(AIXOpLayer, self).__init__()
        self.attrs = {"aix_graph_path": path}
        self.name = "AixOp"
        self.op = "AixOp"
        self.inputs = None
        self.outputs = outputs


    def forward(self, x):
        """Behaviour of custom operation."""

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
    """
    A class responsible for processing a frozen PyTorch model, integrating AIX graph
    operations, and managing hook outputs during the forward pass.
    
    Attributes:
        frozen_model_path (str): Path to the pre-trained and frozen model.
        aix_graph_path (str): Path to the AIX graph file.
        ir_graph (AxfcIRGraph): An AIX graph instance used for writing the launcher.
        md (AxfcMachineDesc): A machine description instance.
        model (torch.nn.Module): The loaded PyTorch model.
        gm (torch.fx.GraphModule): A GraphModule obtained from symbolic tracing of `model`.
        hook_outputs (dict): Stores outputs from the registered hooks during the forward pass.
    """


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
    
    
    def hook(self, module, input, output):
        """
        Hook function to capture the output of a module during the forward pass.
        
        Args:
            module: The module for which the output is being captured.
            input: The input tensor to the module.
            output: The output tensor from the module.
        """
        module_key = module.__class__.__name__
        self.hook_outputs[module_key] = output.detach()
    

    
    def get_custom_graph(self):
        """
        Modifies the model's computational graph by integrating custom AIX operations and
        exports the modified model to ONNX format.
        
        Returns:
            A tuple containing the status of the operation and the path to the saved model.
        """

        tensor = self.generate_input_tensor()

        for block in self.ir_graph.blocks:
            # Skip blocks that do not support AIXH conversion.
            if not block.is_aixh_support:
                continue

            # Traverse the IR graph block and find corresponding nodes in the FX graph.
            ir_block_nodes = self.traverse_block(block)
            gr_block_nodes = self.find_cor_graph_nodes(ir_block_nodes)

            # Register a forward hook on the output node of the block to capture its tensor.
            if gr_block_nodes[-1].name == block.output_nodes[0].name:
                submodule = getattr(self.gm, gr_block_nodes[-1].target)
                submodule.register_forward_hook(self.hook)

            self.gm.recompile()
            self.gm(tensor)
        
            # Add new module to the fx graph
            self.gm.add_module('AixOp', AIXOpLayer(
                self.aix_graph_path, list(self.hook_outputs.items())[0][1]
            ))

            # Find the node before new_node
            pred_node = {}
            if 'Input' in block.input_nodes[0].op:
                pred_node = list(self.gm.graph.nodes)[0]
            else:
                for node in list(self.gm.graph.nodes):
                    if not block.input_nodes[0].op == node.op:
                        continue
                    pred_node = node

            # Insert the new AIX operation node after the predecessor.
            with self.gm.graph.inserting_after(pred_node):
                new_node = self.gm.graph.create_node(
                    'call_module',
                    'AixOp',
                    args=(list(self.gm.graph.nodes)[1].args[0],)
                )
            
            # Remove nodes and and rewire dependency
            for node in gr_block_nodes:
                for user, _ in list(node.users.items()):
                    user.replace_input_with(node, new_node)
                self.gm.graph.erase_node(node)

        self.gm.recompile()

        # Convert custom model to onnx format
        torch.onnx.export(self.gm, tensor, 
                          self.get_model_path('onnx'), 
                          opset_version=11,
                          input_names=['data'],
                          output_names=['output'])

        # Save PyTorch custom model
        saved_path = self.get_model_path('pt')
        torch.save(self.gm, saved_path)
        return AxfcError.SUCCESS, saved_path
    

    def generate_input_tensor(self, batch_size = 1):
        """Generate input_shpae of model."""

        first_layer = next(self.model.children())
        if isinstance(first_layer, nn.Conv2d):
            # For Conv2D, the expected input format is (N, C, H, W)
            return torch.randn(batch_size, first_layer.in_channels, 224, 224)
        if isinstance(first_layer, nn.Linear):
            # For Linear, the expected input format is (N, in_features)
            return torch.randn(batch_size, first_layer.in_features)
        else:
            raise TypeError("Unsupported Type")
    

    def traverse_block(self, block):
        """Traverses nodes within a block.

        Args:
            block: AIXH block contains nodes. 
        """
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
    

    def find_cor_graph_nodes(self, nodes_in_block):
        """Find the corresponding node in the FX Graph.

        Args:
            node_in_block: Nodes in the AIXH block of ir_graph.
        """
        corresponding_nodes = []
        for node in self.gm.graph.nodes:
            for b_node in nodes_in_block:
                if node.name == b_node.name:
                    corresponding_nodes.append(node)

        return corresponding_nodes
    

    def get_model_path(self, extension):
        """
        Generates a file path for saving the model with a given extension.
        """
        return self.frozen_model_path.rsplit('.', 1)[0] + f'_custom.{extension}'

