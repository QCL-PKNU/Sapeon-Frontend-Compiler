import torch
import torch.fx
from AxfcIRGraph import *
from AxfcMachineDesc import *

#######################################################################
# AxfcPTWriter class
#######################################################################


class AIXOpLayer(torch.nn.Module):
    def __init__(self, path: str):
        super(AIXOpLayer, self).__init__()
        self.path = path

    def forward(self, *inputs):
        return inputs


class AxfcPTWriter:

    ## @var __ir_graph
    # an AIX graph that will be used for writing the laucher 

    ## @var __frozen_model_path
    # the path to the pre-trained and 'frozen' model

    # @var __aix_graph_path
    # the path of aix graph

    # @var hook_outputs
    # to store outputs from the registered hooks during the forward pass

    # @var hook_handles
    # to keep track of the hooks registered on the model

    ## The constructor
    # def __init(self, ir_graph: str, frozen_model_path: str, aix_graph_path: str):
    #     self.__ir_graph = ir_graph
    #     self.frozen_model_path = frozen_model_path
    #     self.aix_graph_path = aix_graph_path

    ## The constructor
    def __init__(self, 
                ir_graph: AxfcIRGraph, 
                frozen_model_path: str, 
                aix_graph_path: str):
        self.ir_graph = ir_graph
        self.frozen_model_path = frozen_model_path
        self.aix_graph_path = aix_graph_path
        self.traced_module: torch.fx.GraphModule = None
        self.hook_outputs = {}
        self.hook_handles = {}

    ## Hook method to capture the output of specific layers during forward pass.
    def hook(self, module, input, output):
        self.hook_outputs[module.name] = output.detach()

    ## This method is used to generate custom model
    #
    # @param self this object
    def get_custom_model(self):
        pt_model = torch.load(self.frozen_model_path)
        self.traced_module = torch.fx.symbolic_trace(pt_model)

        # Iterating through custom operation blocks in the IR graph and replacing them in the model.
        for _, block in enumerate(self.ir_graph.blocks):
            if not block.is_aixh_support:
                continue
            custom_op_layer = AIXOpLayer(f"{self.aix_graph_path}/{block.id}")
            custom_op_name = f"AixOp_{block.id}"
            self.traced_module.add_module(custom_op_name, custom_op_layer)

            nodes_in_block = self.traverse_block(block)
            nodes_to_remove = self.find_corresponding_graph_nodes(nodes_in_block)
            self.replace_with_custom_operator(block, nodes_to_remove)

        # Register forward hooks to capture output from custom layers.
        for node in self.traced_module.graph.nodes:
            if not self.should_register_hook(node):
                continue
            layer_to_replace = self.traced_module.get_submodule(node.name)
            handle = layer_to_replace.register_forward_hook(self.hook)
            self.hook_handles[node] = handle

        # Perform a dummy forward pass to capture custom operation outputs.
        dummy_input = torch.rand(1, 3, 224, 224)
        _ = self.traced_module(dummy_input)

        self.remove_hook()
        self.validate_graph()

        custom_model = self.create_new_model(self.traced_module, self.traced_module.graph)
        saved_path = self.save_custom_model(custom_model)

        return saved_path

    def should_register_hook(self, node):
        return isinstance(node, torch.fx.Node) and node.op == "call_module"

    ## This method is used to traverse nodes within a block and return a set of nodes.
    #
    # @param self this object
    # @param block containing nodes to be traversed.
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

        return nodes_in_block

    ## This method is used to find nodes in the traced module's graph corresponding to nodes in a block.
    #
    # @param self this object
    # @param nodes_in_block set of nodes within a block.
    def find_corresponding_graph_nodes(self, nodes_in_block):
        corresponding_nodes = []
        for node in self.traced_module.graph.nodes:
            if node.name in nodes_in_block:
                corresponding_nodes.append(node)

        return corresponding_nodes

    ## This method is used to replace specified nodes with custom operators in the traced module's graph.
    #
    # @param self this object
    # @param block ir block containing custom operation information.
    # @param nodes_to_remove list of nodes to be replaced
    def replace_with_custom_operator(self, block: AxfcIRBlock, nodes_to_remove):
        input_nodes = [node for node in block.input_nodes]
        predecessors = self.find_predecessors(self.traced_module.graph, input_nodes)
        last_predecessor = max(predecessors, key=lambda n: n._node_idx)

        with self.traced_module.graph.inserting_after(last_predecessor):
            # Retrieve the immediate tensors from the hook outputs
            custom_op_inputs = tuple(
                self.hook_outputs[node.name] for node in predecessors
            )

            # Call the custom module with inputs
            custom_op = self.traced_module.graph.call_module(
                f"AixOp_{block.id}", args=custom_op_inputs
            )

        for node in nodes_to_remove:
            for user in list(node.users):
                # Replace the input of the user nodes from the old node to the custom op node
                user.args = tuple(
                    custom_op if arg is node else arg for arg in user.args
                )

            self.traced_module.graph.erase_node(node)

    ##  This method is used to find predecessor nodes of a set of target nodes in the traced module's graph.
    #
    # @param self this object
    # @param input_nodes nodes of block that take input from outside
    def find_predecessors(self, input_nodes: set):
        predecessors = {
            node
            for node in self.traced_module.graph.nodes
            if any(arg in input_nodes for arg in node.args)
        }

        return predecessors

    ## This method is used to remove forward hooks from layers.
    #
    # @param self this object
    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()

    ## This method is used to recompile and lint the traced module's graph for correctness.
    #
    # @param self this object
    def validate_graph(self):
        self.traced_module.recompile()
        self.traced_module.graph.lint()

    ## This method is used to create a new model with the modified graph.
    #
    # @param self this object
    def create_new_model(self):
        return torch.fx.GraphModule(self.traced_module, self.traced_module.graph)

    ## This method is used to save the custom model to a file and return the file path.
    #
    # @param self this object
    # @param custom_model a custom PyTorch model to be saved
    def save_custom_model(self, custom_model):
        saved_path = self.frozen_model_path.rsplit(".pt", 1)[0] + "_custom.pt"
        torch.save(custom_model, saved_path)
        return saved_path
    