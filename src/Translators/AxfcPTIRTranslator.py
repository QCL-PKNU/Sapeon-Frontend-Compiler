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
#   [Before:High Performance Computing Laboratory (hpcl.pknu.ac.kr)]
#######################################################################


import torch
import torch.fx as fx
import numpy as np

from collections import OrderedDict
from .AxfcIRTranslator import *

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
DUMMY_INPUT = torch.randn((1, 3, 244, 244))
DEFAULT_TYPE = 'NCHW' # Follow the darknet format
DEFAULT_DTYPE = torch.float32 # Default dtype


class AxfcPTIRTranslator(AxfcIRTranslator):
    """PyTorch IR Translator class.

    Args:
        md: Machine description object.
        model_path: Path to the PyTorch model file.

    Attributes:
        _gm: PyTorch graph module.
        _module_symtab: A symbolic table for named modules of Pytorch model.
        _input_names: Input name of model.
    """


    def __init__(self, md, model_path: str):
        super().__init__(md)
        
        model = torch.load(model_path)
        self._gm: fx.graph_module = fx.symbolic_trace(model, (DUMMY_INPUT, ))
        self._pt_graph: fx.graph = self._gm.graph
        # Tensor with input values; weight, bias, mean, etc
        self._tensor_symtab = OrderedDict(self._gm.state_dict())

        # Create symtab for named_module
        self._module_symtab = self.__build_module_symtab(self._gm)
        self._input_names = [node.name for node in self._pt_graph.nodes
                             if node.op in ('placeholder', 'get_attr')]
        
        self._input_tensor_cache = {}
        self._emitted_tensors_cache = {}
    
    
    def __build_module_symtab(self, graph_module):
        """Builds a symbolic table for IR node from Pytorch model."""
        symtab = dict()
        for module_name, attr in graph_module.named_modules():
            symtab[module_name] = attr
                
        return symtab

    
    ##################################### 
    ## emission methods for AIX layers ##
    #####################################
    

    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """Emit the AIXTensor to be the input, output, scale, filter, biases, and variance.

        Args:
            ir_node (AxfcIRNode): The IR node to be emitted as an AIX tensors.

        Returns:
            AIXLayer.AIXTensor: The emitted AIXTensor representing the node's inputs.
        """

        input_nodes = [node for node in ir_node.preds]
        # input_nodes = list(filter(lambda x: x.op != "Const" or x.name in self._input_names, ir_node.preds))
        
        aix_tensors = []
        for pred in input_nodes:
            node_id = pred.node_def.target

            if node_id not in self._input_tensor_cache:
                aix_tensor = self.__emit_aix_tensor(pred)
                self._input_tensor_cache[node_id] = aix_tensor
            else:
                aix_tensor = self._input_tensor_cache[node_id]

            aix_tensors.append(aix_tensor)

        return aix_tensor
    
    
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """Emits an AIX tensor of an output type for the given IR node.

        Args:
            ir_node (AxfcIRNode): The IR node from which to emit the output tensor.

        Returns:
            AIXLayer.AIXTensor: The emitted AIXTensor representing the node's output.
        """
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


    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits a PyTorch-specific convolution layer for the given IR node.
        The information includes layer inputs, layer outputs, and so on.

        Args:
            ir_node (AxfcIRNode): The IR node to be emitted as a convolution layer.
        
        Returns:
            AxfcError: Success or failure of the operation.
        """
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


    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """Emits PyTorch-specific information for a batch normalization layer from the given IR node.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX batchnorm layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # Node inputs such as bias, weight, mean and variance in Tensor format
        aix_layer.bias.CopyFrom(self._emit_aix_tensor_generic(ir_node, "bias"))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_generic(ir_node, "mean"))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_generic(ir_node, "variance"))

        # Attributes for BatchNormalization in float format
        # epsilon, momentum
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS
    

    def _emit_aix_layer_downsample(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """Emits PyTorch-specific information for a downsampling layer from the given IR node
        into the given AIX downsample layer object.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX downsample layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_downsample - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS

    
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """Emits PyTorch-specific details into an AIX element-wise add (EWAdd) layer object for a given IR node.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.
        
        Returns:
            AxfcError: Status of the operation.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        ewadddesc = AIXLayer.AIXEWAddDesc()
        scale_size = len(ir_node.preds)

        ewadddesc.scale.extend([1] * scale_size)

        aix_layer.ewadddesc.CopyFrom(ewadddesc)

        return AxfcError.SUCCESS

    
    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits PyTorch-specific information into the given AIX maxpool layer object
        from an IR node, including layer inputs and outputs.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX maxpool layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        #filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=[]))

        # samplingdesc
        # dilation, kernel_size, padding, stride
        aix_layer.samplingdesc.CopyFrom(self._emit_aix_sampling_desc(ir_node))

        return AxfcError.SUCCESS


    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits PyTorch-specific information into the given AIX avgpool layer object
        from an IR node, including layer inputs and outputs.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX avgpool layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        sampling_desc = self._emit_aix_sampling_desc(ir_node)

        sampling_desc.stride[:] = []
        sampling_desc.stride.extend([0, 0, 0, 0])

        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS


    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits PyTorch-specific information into the given AIX activation layer object
        from an IR node. Information includes layer inputs, outputs, and potentially
        other parameters like epsilon and momentum.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX activation layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_activation - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        # epsilon, momentum
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))

        return AxfcError.SUCCESS
    
    
    def _emit_aix_layer_wildcard(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """Emits the given IR node into the given AIX wildcard layer object. 
        The information includes layer inputs, outputs, and so on.

        Args:
            ir_node (AxfcIRNode): The IR node to be processed.

        Returns:
            An output AIX wildcard layer.
        """
        logging.info("AxfcPTBuilderTranslator:_emit_aix_layer_activation - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node))
        
        return AxfcError.SUCCESS
    
    
    ######################################
    ## emission methods for AIX tensors ##
    ######################################

    def __emit_aix_tensor(self, node: [AxfcIRNode, str], is_inout_tensor=False) -> AIXLayer.AIXTensor:
        """Emit an AIX Tensors to be the input, output, scale, filter, biase, and variance.

        Args:
            node (AxfcIRNode or str): An IR node or tensor name to emit the AIX tensor from.
            is_inout_tensor (bool): Indicates if the tensor is an input/output tensor.

        Returns:
            AIXLayer.AIXTensor: The emitted AIX Tensor object.
        """
        # Simplify cache key to ensure it captures both the node and its in/out state
        cache_key = f"{node}_{is_inout_tensor}"
        if cache_key in self._emitted_tensors_cache:
            return self._emitted_tensors_cache[cache_key]

        aix_tensor = AIXLayer.AIXTensor()
        exception_op = ['add']
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
            dtype = self.DEFAULT_DTYPE

        aix_tensor.dtype = aix_data_type_tbl.get(dtype)

        if dtype is None:
            data_format = DEFAULT_TYPE.encode()
        elif len(tensor.shape) == 1:
            data_format = b'VECTOR'
        else:
            data_format = DEFAULT_TYPE.encode()

        # Set format
        aix_tensor.format = aix_tensor_format_tbl[data_format]

        # NOTE If requires_grad is False, it means constant which is not traced by autograd machine
        if tensor.requires_grad == False:
            tensor_values = torch.flatten(tensor)
            for fval in tensor_values:
                aix_tensor.fval.append(fval.item())

        if tensor.shape:
            shape = list(map(lambda x: 1 if not x else x, tensor.shape))
                
            # AIXGraph shape is NCHW, in reversed order
            shape.reverse()
            aix_tensor.dims.extend(shape)
        else:
            logging.warning(f"AxfcPyTorchIRTranslator: {tensor_name} shape is invalid.")

        # Compute and set tensor size
        aix_tensor.size = int(np.prod(aix_tensor.dims))

        self._emitted_tensors_cache[cache_key] = aix_tensor
        return aix_tensor


    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """Emits an AIX tensor of a filter type from the given IR node.

        Args:
            ir_node (AxfcIRNode): An IR node to be emitted as an AIX tensor.
            kwargs: Additional keyword arguments, potentially including 'tensor'.

        Returns:
            AIXLayer.AIXTensor: An AIX tensor of a filter type.
        """
        tensor_name = ir_node.node_def.target
        if tensor_name + ".weight" in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(f"{tensor_name}.weight")
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
        else:
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
    

    def _emit_aix_tensor_generic(self, ir_node: AxfcIRNode, tensor_type: str) -> AIXLayer.AIXTensor:
        """Emits an AIX tensor bias, mean or variance type from the given IR node.
        
        Args:
            ir_node: An IR node to be emitted as an AIX tensor.
            kwargs: Keyword arguments used for passing the tensor.

        Returns:
            AIXLayer.AIXTensor: An AIX tensor of bias, mean or variance type.
        """
        tensor_suffix_map = {
            "bias": ".bias",
            "mean": ".running_mean",
            "variance": ".running_var"
        }

        tensor_suffix = tensor_suffix_map.get(tensor_type, "")
        tensor_name = ir_node.node_def.target + tensor_suffix

        if tensor_name in self._tensor_symtab.keys():
            aix_tensor = self.__emit_aix_tensor(tensor_name)
        else:
            # Fallback to a default tensor emission if specific tensor not found
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, default_value=1)

        return aix_tensor
    
    
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """Emits the AIX convolution description from a given IR node.

        Args:
            ir_node (AxfcIRNode): An IR node to be emitted as an AIX tensor.
            kwargs: Keyword arguments used for pass the 'tensor'

        Returns:
            AIXLayer.AIXConvolutionDesc: The convolution description for the AIX layer.
        """
        # Get convolution layer description from 'aixh_pb2.py'
        convolution_desc = AIXLayer.AIXConvolutionDesc()
        convolution_desc.dtype = ir_node.aix_layer.input.dtype

        # Retrieve the PyTorch module from the symbol table
        pt_node = self._module_symtab[ir_node.node_def.target]

        # Helper function to simplify setting attributes
        def get_attr_with_default(attr, default):
            return getattr(pt_node, attr, default)

        # Set stride, padding, dilation, and groups with fallbacks to defaults if not specified
        stride = get_attr_with_default("stride", (1, 1))
        padding = get_attr_with_default("padding", (0, 0))
        dilation = get_attr_with_default("dilation", (1, 1))
        groups = get_attr_with_default("groups", 1)

        # Update the convolution descriptor
        convolution_desc.stride.extend([stride[0], stride[1], 0, 0])
        convolution_desc.padding.extend([padding[0], padding[1], 0, 0])
        convolution_desc.dilation.extend([dilation[0], dilation[1], 0, 0])
        convolution_desc.groups = groups

        return convolution_desc

    
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits the AIX sampling description from a given IR node.
        This method must called after _emit_aix_convolution_desc()

        Args:
            ir_node (AxfcIRNode): The IR node to be emitted as an AIX tensor.

        Returns:
            AIXLayer.AIXSamplingDesc: The sampling description for the AIX layer.
        """

        # Retrieve sampling layer description from 'aixh_pb2.py'        
        sampling_desc = AIXLayer.AIXSamplingDesc()

        # Retrieve the PyTorch module from the symbol table
        pt_node = self._module_symtab[ir_node.node_def.target]

        # if kwargs['tensor']:
        #     tensor = kwargs['tensor']
        # else:
        #     logging.error("AxfcPTIRTranslator:_emit_aix_sampling_desc - need TensorProto object")

        # Define a priority for layer types if a node can have multiple types

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
    
    
    def __emit_default_hyper_parameter(self, aix_layer: AIXLayer, default_value: int) -> AIXLayer.AIXTensor:
        """Creates an AIXTensor with default hyperparameters: scale, mean, variance

        Args:
            aix_layer (AIXLayer): The AIXLayer object to which the tensor belongs.
            default_value (int): The default value to set for each element in the tensor.

        Returns:
            AIXLayer.AIXTensor: A tensor that can be mean, scale, or variance tensor
        """
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