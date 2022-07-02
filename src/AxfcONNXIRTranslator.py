#######################################################################
#   AxfcONNXIRTranslator
#
#   Created: 2022. 02. 10
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Hour Leanghok (leangyok@pukyong.ac.kr)
#
#   Quantum Computing Laboratory (quantum.pknu.ac.kr)
#######################################################################

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

import tensorflow as tf
import numpy as np

from AxfcIRTranslator import *
from util import loadFrozenModel, print_tensor_content

#######################################################################
# Global tables for AIXDataType and AIXTensorFormat
#######################################################################

## AIXDataType table
# onnx_surgeon uses numpy datatype which can be represent using string or np.float32(ex)
aix_data_type_tbl = {
    'half': AIXLayer.AIXDataType.AIX_DATA_HALF,
    'float16': AIXLayer.AIXDataType.AIX_DATA_HALF,
    'float32': AIXLayer.AIXDataType.AIX_DATA_FLOAT,
    'float64': AIXLayer.AIXDataType.AIX_DATA_DOUBLE,
    'uint8': AIXLayer.AIXDataType.AIX_DATA_UINT8,
    'int8': AIXLayer.AIXDataType.AIX_DATA_SINT8,
    'int16': AIXLayer.AIXDataType.AIX_DATA_SINT16

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
DEFAULT_DTYPE = 'float32'

#######################################################################
# AxfcTFIRTranslator class
#######################################################################

class AxfcONNXIRTranslator(AxfcIRTranslator):
    

    ## The constructure
    def __init__(self, md, path):
        super().__init__(md)

        onnx_model = onnx.load(path)
        inferred_model = shape_inference.infer_shapes(onnx_model)
        # self.graph = gs.import_onnx(onnx_model)
        self.graph = gs.import_onnx(inferred_model)
        self.tensors = self.graph.tensors() #orderedDict of tensors
        self._symtab = self.__build_node_symtab(self.graph) #build name:node symtab

        #TESTING
        # for k,v in self.tensors.items():
        #     print(v.dtype, str(k))

    #build name:node symtab
    def __build_node_symtab(self, onnx_graph):
        symtab = dict()
        for node in onnx_graph.nodes:
            symtab[node.name] = node

        return symtab
    
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_input(ir_node, **kwargs)

        #TESTING emitting current node tensor
        # return self.__emit_aix_tensor(ir_node)

        #get input tensor
        #TESTING on data as input
        input_nodes = list(filter(lambda x: x.op != "Const" or x.name == "data", ir_node.preds))
        input_tensors = [self.tensors[node.name] for node in input_nodes]

        aix_tensors = []
        for input_tensor in input_tensors:
            aix_tensor = self.__emit_aix_tensor(input_tensor, is_inout_tensor=True)
            aix_tensors.append(aix_tensor)

        return aix_tensor

        #return should be a list of aix_tensors from input
        return aix_tensors

    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_output(ir_node, **kwargs)

        aix_tensor = self.__emit_aix_tensor(ir_node)
        return aix_tensor

    def __emit_aix_tensor(self, ir_node, is_inout_tensor=False, **kwargs) -> AIXLayer.AIXTensor:
        aix_tensor = AIXLayer.AIXTensor()
        
        #get tensor
        tensor = self.tensors[ir_node.name]

        
        dtype = tensor.dtype

        if not dtype:
            dtype = DEFAULT_DTYPE
        
        #TESTING dtype always float32
        aix_tensor.dtype = 0
        # aix_tensor.dtype = aix_data_type_tbl[str(dtype)]

        #onnx data format is default at NCHW if any dtype is present then it is a vector
        if dtype is None:
            data_format = DEFAULT_TYPE.encode()
        else:
            data_format = b'VECTOR'
        
        #set format
        aix_tensor.format = aix_tensor_format_tbl[data_format]

        #set aix tensor fval
        #if tensor is a constant then fval need to be set
        if type(tensor) is gs.ir.tensor.Constant:
            tensor_values = tensor.values.flatten()
            for dim in tensor_values:
                aix_tensor.fval.append(dim)

        #set aix tensor dims
        if tensor.shape:
            shape = list(map(lambda x: 1 if not x else x, tensor.shape))
            aix_tensor.dims.extend(shape)

        else:
            logging.warning("AxfcONNXIRTranslator: shape is none, should have default value or validate when this happens.")

        #set aix tensor size
        aix_tensor.size = int(np.prod(aix_tensor.dims))

        return aix_tensor
    
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        # return super()._emit_aix_layer_convolution(ir_node, **kwargs)
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.name]
        onnx_node = self._symtab[ir_node.name]

        #filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        #bias
        if "bias" in onnx_node.attrs:
            aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=tensor))

        # # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']

        return AxfcError.SUCCESS

    
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_filter(ir_node, **kwargs)

        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if 'weight' in input.name:
                aix_tensor = self.__emit_aix_tensor(input)
        
        if aix_tensor is None:
            aix_layer = ir_node.aix_layer
            aix_tensor = AIXLayer.AIXTensor()
            aix_tensor.dtype = aix_layer.input.dtype
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
            aix_tensor.dims.append(1)  # width
            aix_tensor.dims.append(1)  # height

            input_dims_dict = self.__get_aix_tensor_dims(aix_layer.input)
            output_dims_dict = self.__get_aix_tensor_dims(aix_layer.output)
            aix_tensor.dims.append(input_dims_dict['C'])  # channel
            aix_tensor.dims.append(output_dims_dict['C'])  # number of filter
        
        return aix_tensor
    
    ##  This method get aixtensor dims from format as dictionary
    #
    # @param self this object
    # @param AIXTensor an an AIX tensor data contains dims, data format, dtype, size and ptr
    # @return a dictionary object has key as element of data format. e.g input['H'] = 2
    def __get_aix_tensor_dims(self, aix_tensor: AIXLayer.AIXTensor) -> dict:

        return self.__get_values_of_format(aix_tensor.dims, aix_tensor.format)
    
    ##  This method get data from aix_tensor_format format as dictionary
    #
    # @param self this object
    # @param values an list of input values
    # @param tensor_format an AIX tensor format
    # @return a dictionary object has key as element of data format. e.g input['H'] = 2
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

    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_scale(ir_node, **kwargs)

        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if 'gamma' in input.name:
                aix_tensor = self.__emit_aix_tensor(input)
        
        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    
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

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_scale(ir_node, **kwargs)

        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if 'running_mean' in input.name:
                aix_tensor = self.__emit_aix_tensor(input)
        
        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    
    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_scale(ir_node, **kwargs)

        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if 'bias' in input.name:
                aix_tensor = self.__emit_aix_tensor(input)
        
        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX batchnorm layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.name]
        onnx_node = self._symtab[ir_node.name]

        #filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.scale.CopyFrom(self._emit_aix_tensor_scale(ir_node, tensor=tensor))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node, tensor=tensor))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node, tensor=tensor))

        # # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']

        return AxfcError.SUCCESS

    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # return super()._emit_aix_tensor_scale(ir_node, **kwargs)

        node_inputs = ir_node.preds

        aix_tensor = None

        for input in node_inputs:
            if 'running_var' in input.name:
                aix_tensor = self.__emit_aix_tensor(input)
        
        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_activation - node %d, %s",
                     ir_node.layer_id, ir_node.op)
        
        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.name]
        onnx_node = self._symtab[ir_node.name]

        # filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']
        
        return AxfcError.SUCCESS

    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.name]
        onnx_node = self._symtab[ir_node.name]

        # filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']

        # samplingdesc
        # sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)
        # aix_layer.samplingdesc.CopyFrom(sampling_desc)
        
        return AxfcError.SUCCESS
    
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.name]
        onnx_node = self._symtab[ir_node.name]

        # # filter
        # aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        ewadddesc = AIXLayer.AIXEWAddDesc()
        scale_size = len(onnx_node.inputs)

        ewadddesc.scale.extend([1] * scale_size)

        aix_layer.ewadddesc.CopyFrom(ewadddesc)

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']
        
        return AxfcError.SUCCESS
    
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXConvolutionDesc:
        
        onnx_node = self._symtab[ir_node.name]

        convolution_desc = AIXLayer.AIXConvolutionDesc()

        aix_layer = ir_node.aix_layer

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        #strides
        if "strides" in onnx_node.attrs:
            # print(onnx_node.attrs["strides"])
            stride_dict = dict(zip('HW', onnx_node.attrs["strides"]))
            convolution_desc.stride.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            convolution_desc.stride.extend([1,1,0,0])
        
        #pads
        if 'pads' in onnx_node.attrs:
            convolution_desc.padding.extend(onnx_node.attrs["pads"])
        else:
            convolution_desc.padding.extend([0, 0, 0, 0])
        
        # dilations
        if 'dilations' in onnx_node.attrs:
            convolution_desc.dilation.extend(onnx_node.attrs['dilations'])
            convolution_desc.dilation.extend([1,1])
        else:
            convolution_desc.dilation.extend([1,1,1,1])
        
        # group
        if 'group' in onnx_node.attrs:
            convolution_desc.groups = onnx_node.attrs['group']
        else:
            convolution_desc.groups = 1
        
        return convolution_desc





