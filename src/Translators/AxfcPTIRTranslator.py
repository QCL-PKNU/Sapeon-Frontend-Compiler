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
import torch.fx

import numpy as np
from collections import OrderedDict
# from ..AxfcIRGraph import *

from .AxfcIRTranslator import *
# from ..AxfcMachineDesc import AxfcError

#######################################################################
# Global tables for AIXDataType and AIXTensorFormat
#######################################################################

## AIXDataType table
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type

aix_data_type_tbl = {
    torch.float16: AIXLayer.AIXDataType.AIX_DATA_HALF,
    torch.float32: AIXLayer.AIXDataType.AIX_DATA_FLOAT,
    torch.float64: AIXLayer.AIXDataType.AIX_DATA_DOUBLE,
    torch.uint8: AIXLayer.AIXDataType.AIX_DATA_UINT8,
    torch.int8: AIXLayer.AIXDataType.AIX_DATA_SINT8,
    torch.int16: AIXLayer.AIXDataType.AIX_DATA_SINT16
}

## AIXTensorFormat table
aix_tensor_format_tbl = {
    b"NCHW": AIXLayer.AIXTensorFormat.AIX_FORMAT_NCHW,
    b"NHWC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NHWC,
    b"NWHC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NWHC,
    b"VECTOR": AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR
}

#######################################################################
# Global variable
#######################################################################

# following the darknet format
DEFAULT_TYPE = 'NCHW'

#Default dtype
DEFAULT_DTYPE = torch.float32

# NOTE
# ** Sanghyeon - symbolic_traced model doens't have a real input data tensor
# @torch.fx.wrap
def my_randn(shape):
    return torch.randn(shape)

DUMMY_INPUT = my_randn((1, 3, 244, 244))

#######################################################################
# AxfcPTIRTranslator class
#######################################################################

class AxfcPTIRTranslator(AxfcIRTranslator):

    ## @var _pt_model
    # pytorch model

    ## @var tensor_symtab
    # tensor sets of pytorch graph

    ## @var _module_symtab
    # symbolic table for named modules of pytorch model

    ## @var _input_names
    # input names of model

    ## The constructor
    def __init__(self, md, model_path: str):
        super().__init__(md)

        # To load complete torch.graph, need to load module and state together
        pt_model: torch.nn.Module = torch.load(model_path)
        # pt_model.eval()

        # ** Sanghyeon - symbolic_trace makes all node as constant
        # self._pt_model : torch.fx.graph_module = torch.fx.symbolic_trace(pt_model, (DUMMY_INPUT, ))
        self._pt_model : torch.fx.graph_module = torch.fx.symbolic_trace(pt_model, (DUMMY_INPUT, ))
       

        self._pt_graph: torch.fx.graph = self._pt_model.graph
        self._tensor_symtab: OrderedDict = self._pt_model.state_dict() # tensors has the inputs value such as weight, bias, mean, etc
        self._module_symtab = self.__build_module_symtab(self._pt_model) # make symtab for named_modules
        self._input_names = [node.name for node in self._pt_graph.nodes 
                             if node.op == 'placeholder' or node.op == "get_attr"] # the input for model has name of 'placeholder'
    
    ## This method is used to build symbolic table of IR node.
    #
    # @param self this object
    # @param pt_model pytorch model 
    # @return symbolic table
    def __build_module_symtab(self, pt_model):
        symtab = dict()
        for module_name, attr in pt_model.named_modules():
            symtab[module_name] = attr
                
        return symtab

    
    ##################################### 
    ## emission methods for AIX layers ##
    #####################################
    
    ## This method is used to emit the AIXTensor to be the input, output, scale, filter, biases, and variance.
    #  
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensors
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        #extract input tensor of torch.graph
        #the input of torch.grpah named as 'placeholder'
        input_nodes = [node for node in ir_node.preds]
        # input_nodes = list(filter(lambda x: x.op != "Const" or x.name in self._input_names, ir_node.preds))
        
        aix_tensors = []
        for input_node in input_nodes:
            # tensor_name = input_node.node_def.target
            aix_tensor = self.__emit_aix_tensor(input_node)
            aix_tensors.append(aix_tensor)

        return aix_tensor
    
    ## This method is used to emit an AIX Tensors of an output type from the given IR node.
    #  
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # Read the output nodes of IR_node
        output_nodes = [node for node in ir_node.succs]
        # output_nodes = list(lambda x: x.op != "Const" or x.name in ir_node.succs)
        
        aix_tensors = []
        # for output_node in output_nodes:
        #     tensor_name = output_node.node_def.target
        #     aix_tensor = self.__emit_aix_tensor(output_node, is_inout_tensor=True)
        #     aix_tensors.append(aix_tensor)

       
        # Emit tensor
        aix_tensor = self.__emit_aix_tensor(ir_node)

        return aix_tensor

    ## This method emits some pytorch-specific information of the given IR node
    # into the given AIX convolution layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX convolution layer
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # filter (== weight)
        # load filter tensor from tensors
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node))

        # bias
        # aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node))
        
        # convolution layer attrtibutes description
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS


    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX batchnorm layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # Node inputs such as bias, weight, mean and variance in Tensor format
        aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node))

        # Attributes for BatchNormalization in float format
        # epsilon, momentum
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS
    

    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX downsample layer object.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX downsample layer
    def _emit_aix_layer_downsample(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_downsample - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS

    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX element-wise add (ewadd) layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        ewadddesc = AIXLayer.AIXEWAddDesc()
        scale_size = len(ir_node.preds)

        ewadddesc.scale.extend([1] * scale_size)

        aix_layer.ewadddesc.CopyFrom(ewadddesc)

        return AxfcError.SUCCESS

    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX maxpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX maxpool layer
    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        #filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=[]))

        # samplingdesc
        # dilation, kernel_size, padding, stride
        aix_layer.samplingdesc.CopyFrom(self._emit_aix_sampling_desc(ir_node))

        return AxfcError.SUCCESS

    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX avgpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        sampling_desc = self._emit_aix_sampling_desc(ir_node)

        sampling_desc.stride[:] = []
        sampling_desc.stride.extend([0, 0, 0, 0])

        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS

    ##  This method emits some pytorch-specific information of the given IR node
    # into the given AIX activation layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX activation layer
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_activation - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # epsilon, momentum
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS
    
    ## This method emits the given IR node into the given AIX wildcard layer object.
    # The information includes layer inputs, outputs, and so on.
    # 
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX wildcard layer
    def _emit_aix_layer_wildcard(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_activation - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))
        
        return AxfcError.SUCCESS
    
    
    ######################################
    ## emission methods for AIX tensors ##
    ######################################

    ## This method is used to emit an AIX Tensors to be the input, output, scale, filter, biase, and variance.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @is_inout_tensor in/output tensor checker
    # @return aix_tensor AIXTensor object
    def __emit_aix_tensor(self, node: [AxfcIRNode, str], is_inout_tensor=False) -> AIXLayer.AIXTensor:
            # Initialize AIXLayer
            aix_tensor = AIXLayer.AIXTensor()

            exception_op = ['add']
            # exception_op = ['relu', 'maxpool', 'avgpool', 'flatten']

            if isinstance(node, AxfcIRNode) and node.op == "Input":
                tensor_name = node.node_def.target
            elif isinstance(node, AxfcIRNode) and node.op in exception_op:
                tensor_name = node.name
            elif isinstance(node, AxfcIRNode):
                tensor_name = node.node_def.target + ".weight"
            else:
                tensor_name = node

            # Get tensor
            if self._tensor_symtab.get(tensor_name) is None:
                tensor = self._tensor_symtab['_tensor_constant0']                
            else:
                tensor = self._tensor_symtab[tensor_name]

            dtype = tensor.dtype

            if not dtype:
                dtype = DEFAULT_DTYPE

            aix_tensor.dtype = aix_data_type_tbl.get(dtype)

            if dtype is None:
                data_format = DEFAULT_TYPE.encode()
            elif len(tensor.shape) == 1:
                data_format = b'VECTOR'
            else:
                data_format = DEFAULT_TYPE.encode()

            #set format
            aix_tensor.format = aix_tensor_format_tbl[data_format]

            # NOTE If requires_grad is False, it means constant which is not traced by autograd machine
            if tensor.requires_grad == False:
                tensor_values = torch.flatten(tensor)
                for fval in tensor_values:
                    aix_tensor.fval.append(fval)

            if tensor.shape:
                shape = list(map(lambda x: 1 if not x else x, tensor.shape))
                
                #aixgraph shape is NCHW, in reversed order
                shape.reverse()
                aix_tensor.dims.extend(shape)
            else:
                logging.warning(f"AxfcPyTorchIRTranslator: {tensor_name} shape is invalid.")

            #set aix tensor size
            aix_tensor.size = int(np.prod(aix_tensor.dims))

            return aix_tensor

    ## This method emits an AIX tensor of an filter type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an filter type
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        # node_inputs = ir_node.preds

        # tensor_names = [node.node_def.target for node in node_inputs]

        aix_tensor = None

        tensor_name = ir_node.node_def.target
        if tensor_name + ".weight" in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(f"{tensor_name}.weight")
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']

        if aix_tensor is None:
            aix_layer = ir_node.aix_layer
            aix_tensor = AIXLayer.AIXTensor()
            aix_tensor.dtype = aix_layer.input.dtype
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
            aix_tensor.dims.append(1)
            aix_tensor.dims.append(1)

            input_dims_dict = self.__get_aix_tensor_dims(aix_layer.input)
            output_dims_dict = self.__get_aix_tensor_dims(aix_layer.output)
            aix_tensor.dims.append(input_dims_dict['C'])
            aix_tensor.dims.append(output_dims_dict['C'])

        return aix_tensor
    

    def __get_aix_tensor_dims(self, aix_tensor: AIXLayer.AIXTensor) -> dict:
        return self.__get_values_of_format(aix_tensor.dims, aix_tensor.format)
    
    def __get_values_of_format(self, values: list,
                               tensor_format: AIXLayer.AIXTensorFormat) -> dict:
        
        # query string data format from aix_tensor_format_tbl
        data_format = ([k for k, v in aix_tensor_format_tbl.items()
                        if v == tensor_format])[0].decode()
        
        # Darknet dims tensor is reverse
        reverse_values = list(values)
        reverse_values.reverse()

        # map the data format with its value
        return dict(zip(data_format, reverse_values))

    ##  This method emits an AIX tensor of an bias type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an bias type
    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        aix_tensor = None

        tensor_name = ir_node.node_def.target
        if tensor_name + ".bias" in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(f"{tensor_name}.bias")


        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits an AIX tensor of an mean type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an mean type
    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        aix_tensor = None

        tensor_name = ir_node.node_def.target
        if tensor_name + ".running_mean" in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(f"{tensor_name}.running_mean")

        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)
    
        return aix_tensor

    ##  This method emits an AIX tensor of an variance type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an variance type
    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        aix_tensor = None
        
        # node inputs
        tensor_name = ir_node.node_def.target
        if tensor_name + ".running_var" in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(f"{tensor_name}.running_var")

        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)
    
        return aix_tensor
    
    ##  This method emits the AIX convolution description of the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX convolution description
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:

        # get convolution layer description from 'aixh_pb2.py'
        convolution_desc = AIXLayer.AIXConvolutionDesc()

        aix_layer = ir_node.aix_layer

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        # extract node attributes
        pt_node = self._module_symtab[ir_node.node_def.target]

        # stride
        if "stride" in vars(pt_node):
            stride_dict = dict(zip("HW", pt_node.stride))
            convolution_desc.stride.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            convolution_desc.stride.extend([1, 1, 0, 0])

        # padding
        if "padding" in vars(pt_node):
            convolution_desc.padding.extend(pt_node.padding)
        else:
            convolution_desc.padding.extend([0, 0, 0, 0])

        # dilation
        if "dilation" in vars(pt_node):
            convolution_desc.dilation.extend(pt_node.dilation)
            convolution_desc.dilation.extend([1, 1])
        else:
            convolution_desc.dilation.extend([0, 0, 0, 0])

        # group
        # Conv layer in PT model doesn't have group attributes
        if hasattr(pt_node, "groups" or "group"):
            convolution_desc.groups = getattr(pt_node, "groups")
        else:
            convolution_desc.groups = 1 

        return convolution_desc

    ##  This method emits the AIX sampling description of the given IR node.
    #  This method must called after _emit_aix_convolution_desc()
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX sampling description
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:

        # get sampling layer description from 'aixh_pb2.py'        
        sampling_desc = AIXLayer.AIXSamplingDesc()

        aix_layer = ir_node.aix_layer

        # extract node attributes
        pt_node = self._module_symtab[ir_node.node_def.target]

        # if kwargs['tensor']:
        #     tensor = kwargs['tensor']
        # else:
        #     logging.error("AxfcPTIRTranslator:_emit_aix_sampling_desc - need TensorProto object")

        # mode
        for layer_type in ir_node.aix_layer.type:
            if layer_type == AIXLayer.AIXLayerType.AIX_LAYER_MAXPOOL:
                sampling_desc.mode = AIXLayer.AIXSamplingMode.AIX_POOLING_MAX
            elif layer_type == AIXLayer.AIXLayerType.AIX_LAYER_AVGPOOL:
                sampling_desc.mode = AIXLayer.AIXSamplingMode.AIX_POOLING_AVERAGE
            elif layer_type == AIXLayer.AIXLayerType.AIX_LAYER_REORG:
                sampling_desc.mode = AIXLayer.AIXSamplingMode.AIX_POOLING_REORG
            elif layer_type == AIXLayer.AIXLayerType.AIX_LAYER_UPSAMPLE:
                sampling_desc.mode = AIXLayer.AIXSamplingMode.AIX_POOLING_UPSAMPLE
            elif layer_type == AIXLayer.AIXLayerType.AIX_LAYER_PIXELSHUFFLE:
                sampling_desc.mode = AIXLayer.AIXSamplingMode.AIX_POOLING_PIXELSHUFFLE

            sampling_desc.window.extend([0, 0, 0, 0])

        # stride
        if "stride" in vars(pt_node):
            if isinstance(pt_node.stride, int): # If kernel_size is integer
                stride_size = pt_node.stride 
                sampling_desc.stride.extend([stride_size, stride_size, 0, 0])
            else: # if kernel_size is tuple
                stride_size = dict(zip("HW", pt_node.stride))
                sampling_desc.stride.extend([stride_size['H'], stride_size['W'], 0 ,0])
        else:
            sampling_desc.stride.extend([0, 0, 0, 0])

        # padding
        if "padding" in vars(pt_node):
            if isinstance(pt_node.padding, int): # If kernel_size is integer
                padding_size = pt_node.padding
                sampling_desc.padding.extend([padding_size]*4)
            else: # if kernel_size is tuple
                sampling_desc.padding.extend(pt_node.padding)
        else:
            sampling_desc.padding.extend([0, 0, 0, 0])

        return sampling_desc
    
    ## This method is used to set the default hyperparameters: scale, mean, variance
    #
    # @param self this object
    # @param layer AIXLayer object
    # @return AIXTensor it can be mean, scale, or variance tensor
    def __emit_default_hyper_parameter(self, aix_layer: AIXLayer, default_value: int) -> AIXLayer.AIXTensor:
        tensor = AIXLayer.AIXTensor()

        tensor.dtype = aix_layer.output.dtype
        tensor.format = AIXLayer.AIX_FORMAT_VECTOR

        # set the output channel to tensor dims [2]
        output_dims_dict = self.__get_aix_tensor_dims(aix_layer.output)
        output_channel = output_dims_dict['C']
        tensor.dims.append(output_channel)
        tensor.size = output_channel
        tensor.fval.extend([default_value] * output_channel)

        return tensor