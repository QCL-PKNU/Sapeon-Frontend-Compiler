import torch
import torch.fx

import numpy as np
from collections import OrderedDict
from AxfcIRGraph import AxfcError, AxfcIRNode

from AxfcIRTranslator import *
from AxfcMachineDesc import AxfcError

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

#TODO: Clarify the filter usage in pytorch
#TODO: Add the operator info
class AxfcPTIRTranslator(AxfcIRTranslator):

    ## The constructor
    def __init__(self, md, model_path: str, state_path: str):
        super().__init__(md)

        # To load complete torch.graph, need to load module and state together
        pt_model = torch.load(model_path)
        pt_model.load_state_dict(torch.load(state_path))
        pt_model.eval()

        self._pt_model : torch.nn.Module = pt_model
        self.graph: torch.fx.graph = torch.fx.symbolic_trace(pt_model).graph
        # self.tensors: OrderedDict = pt_model.state_dict() # tensors has the inputs value such as weight, bias, mean, etc
        self._symtab = self.__build_node_symtab(self._pt_model) # make symtab for named_modules
        self._input_names = [node.op for node in self.graph.nodes if node.op == 'placeholder'] # the input for model has name of 'placeholder'
        
    def __build_node_symtab(self, pt_model):
        symtab = dict()
        for module in pt_model.named_modules():
            # # replace '_' into '.' / EX) 'layer1_0_conv1' -> 'layer1.0.conv1'
            # node.name = node.name.replace('_', '.')
            # # dictionary symbolic table with {node_name: 'node_name.weight', 'node_name.bias', ...}
            # symtab[node.name] = {key: val for key, val in tensors.items() if key.startswith(node.name)}

            # unpacking the name and attributes
            module_name, module_attr = module

            # save in dict
            symtab[module_name] = module_attr
        
        return symtab
    
    ##################################### 
    ## emission methods for AIX layers ##
    #####################################
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        pt_node = self._symtab[ir_node.name]
        # node_attrs = getattr(self._pt_model, ir_node.name)

        # filter
        # load filter tensor from tensors
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, pt_node[ir_node.name + ".filter"]))

        #bias
        if "bias" in vars(pt_node):
            aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=pt_node.bias))
        
        # convolution layer attrtibutes description
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))


        return AxfcError.SUCCESS


    ## This method emits the given IR node into the given AIX batchnorm layer object.
    # The information includes layer inputs, outputs, and so on.
    # 
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        pt_node = self._symtab[ir_node.name]

        aix_layer = ir_node.aix_layer

        # Node inputs such as bias, weight, mean and variance in Tensor format
        aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=pt_node.bias))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node, tensor=pt_node.running_mean))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node, tensor=pt_node.running_var))

        # Attributes for BatchNormalization in float format
        # epsilon, momentum
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc)

        return AxfcError.SUCCESS

    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        pt_node = self._symtab[ir_node.name]
        
        aix_layer = ir_node.aix_layer

        #filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=[]))

        # samplingdesc
        # dilation, kernel_size, padding, stride
        aix_layer.samplingdesc.CopyFrom(self._emit_aix_sampling_desc(ir_node, tensor=pt_node))

        return AxfcError.SUCCESS

    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        pt_node = self._symtab[ir_node.name]

        aix_layer = ir_node.aix_layer

        #filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=[]))

        # avgpoll desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_sampling_desc(ir_node, tensor=pt_node))

        return AxfcError.SUCCESS

    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()
    
    def _emit_aix_layer_wildcard(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()
    

    ######################################
    ## emission methods for AIX tensors ##
    ######################################
    def __emit_aix_tensor(self, ir_node, is_inout_tensor=False, **kwargs) -> AIXLayer.AIXTensor:
            aix_tensor = AIXLayer.AIXTensor()

            #get tensor
            tensor = self.tensors[ir_node.name]

            dtype = tensor.dtype

            if not dtype:
                dtype = DEFAULT_DTYPE

            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)

            aix_tensor.dtype = aix_data_type_tbl.get(dtype.name)

            if dtype is None:
                data_format = DEFAULT_TYPE.encode()
            elif len(tensor.shape) == 1:
                data_format = b'VECTOR'
            else:
                data_format = DEFAULT_TYPE.encode()

            #set format
            aix_tensor.format = aix_tensor_format_tbl[data_format]

            if type(tensor) is torch.placeholder:
                tensor_values = torch.flatten(tensor)
                for fval in tensor_values:
                    aix_tensor.fval.append(fval)

            if tensor.shape:
                shape = list(map(lambda x: 1 if not x else x, tensor.shape))
                
                #aixgraph shape is NCHW, in reversed order
                shape.reverse()
                aix_tensor.dims.extend(shape)
            else:
                logging.warning(f"AxfcPyTorchIRTranslator: {ir_node.name} shape is invalid.")

            #set aix tensor size
            aix_tensor.size = int(np.prod(aix_tensor.dims))

            return aix_tensor

    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        #extract input tensor of torch.graph
        #the input of torch.grpah named as 'placeholder'
        input_nodes = list(filter(lambda x: x.op != "Const" or x.name in self._input_names, ir_node.preds))
        input_tensors = [self.tensors[node.name] for node in input_nodes]

        aix_tensors = []
        for input_tensor in input_tensors:
            aix_tensor = self.__emit_aix_tensor(input_tensor, is_inout_tensor=True)
            aix_tensors.append(aix_tensor)

        return aix_tensor
     
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        aix_tensor = self.__emit_aix_tensor(ir_node)

        return aix_tensor

    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if "weight" in input.name:  # 'fileter' denotes the weight of node
                aix_tensor = self.__emit_aix_tensor(input)
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

    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if "bias" in input.name:
                aix_tensor = self.__emit_aix_tensor(input)

        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if "running_mean" in input.name:
                aix_tensor = self.__emit_aix_tensor(input)

        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)
    
        return aix_tensor

    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # node inputs
        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if "running_var" in input.name:
                aix_tensor = self.__emit_aix_tensor(input)

        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    
    ## emission methods for AIX convolution dec
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:

        # get convolution layer description from 'aixh_pb2.py'
        convolution_desc = AIXLayer.AIXConvolutionDesc()

        aix_layer = ir_node.aix_layer

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        # extract node attributes
        pt_node = self._symtab[ir_node.name]

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
        if "group" or "groups" in vars(pt_node):
            convolution_desc.groups = getattr(pt_node, "group" or "groups")
        else:
            convolution_desc.groups = 1 

        return convolution_desc

    ## emission methods for AIX sampling dec
    ## For Pooling layer
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:

        # get sampling layer description from 'aixh_pb2.py'        
        sampling_desc = AIXLayer.AIXSamplingDesc()

        aix_layer = ir_node.aix_layer

        # extract node attributes
        pt_node = self._symtab[ir_node.name]

        if kwargs['tensor']:
            tensor = kwargs['tensor']
        else:
            logging.error("AxfcPTIRTranslator:_emit_aix_sampling_desc - need TensorProto object")

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

        ## kernel_shape (kernel_size in PT)
        if "kernel_size" in vars(pt_node):        
            if isinstance(pt_node.kenel_size, int): # If kernel_size is integer
                window_size = pt_node.kernel_size 
                sampling_desc.window.extend([window_size, window_size, 0, 0])

                aix_layer.filter.dims[0] = aix_layer.filter.dims[1] = window_size
            else: # if kernel_size is tuple
                window_size = dict(zip("HW", pt_node.kernel_size))
                sampling_desc.window.extend([window_size['H'], window_size['W'], 0 ,0])
            
                aix_layer.filter.dims[0] = window_size['H']
                aix_layer.filter.dims[1] = window_size['W']
        else:
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
    
    def __emit_default_hyper_parameter(self, aix_layer: AIXLayer, default_value: int) -> AIXLayer.AIXTensor:
        #TODO: Check again code
        tensor = AIXLayer.AIXTensor()

        tensor.dtype = aix_layer.output.dtype
        tensor.format = AIXLayer.AIX_FORMAT_VECTOR

        # set the output channel to tensor dims [2]
        output_dims_dict = self.__get_aix_tensor_dims(aix_layer.output)
        output_channel = output_dims_dict['C']
        tensor.dims.append(output_channel)
        tensor.size = output_channel
        tensor.fval.extend([default_value] * output_channel)

        return 