import torch
import torch.fx
import torchvision


from AxfcIRGraph import *
from AxfcMachineDesc import *

#######################################################################
# AxfcPTWriter class
#######################################################################

DUMMY_INPUT = torch.randn((1, 3, 244, 244))

class AIXOpLayer(torch.nn.Module):
    def __init__(self, path:str):
        super().__init__()
        self.aix_graph_path = path
        self.output_type = torch.float32
        # torch.nn.init.

    def forward(self, x):
        return x


class AxfcPTWriter():
    def __init__(self,
                 ir_graph: AxfcIRGraph,
                 frozen_model_path:str,
                 aix_graph_path:str,
                 kernel_op_path:str,
                 md:AxfcMachineDesc):
        self.__ir_graph = ir_graph
        self.__frozen_model_path = frozen_model_path
        self.__aix_graph_path = aix_graph_path
        self.__kernel_op_path = kernel_op_path
        self.__md = md

    # FIXME: Add function to make custom module following original model's flow
    def replace_with_aixop(self, module, path:str):
        
        customResNet50 = torch.nn.Sequential(
            AIXOpLayer(path),
            module
        )
        
        return customResNet50


    def get_custom_graph(self):
        pt_model: torchvision.models = torch.load(self.__frozen_model_path)

        traced_model: torch.fx.graph = torch.fx.symbolic_trace(pt_model, (DUMMY_INPUT, ))

        pt_tensors: dict = traced_model.state_dict()

        for count, block in enumerate(self.__ir_graph.blocks):
            
            if not block.is_aixh_support:
                continue

            #FIXME: Add function to automatically read input/output module
            inputs = [pt_tensors.get(node.name) for node in block.input_nodes]
            outputs = [node.name for node in block.output_nodes]

            modules = list(filter(lambda name:'avgpool' in name, traced_model.named_modules()))

            # pt_graph.replace_with_aixop(inputs, outputs, self.__aix_graph_path+str(count))
            pt_model = self.replace_with_aixop(modules[0][1], self.__aix_graph_path+str(count))
            

        save_path = self.__frozen_model_path.split(".pt")[0] + "_custom.pt"

        pt_model.save(save_path)

        return AxfcError.SUCCESS, save_path
    
    

