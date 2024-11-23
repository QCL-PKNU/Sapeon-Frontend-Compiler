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

import numpy as np

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from .AxfcIRTranslator import *

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
# AxfcONNXIRTranslator class
#######################################################################

class AxfcONNXIRTranslator(AxfcIRTranslator):
    
    ## @var graph
    # onnx graph

    ## @var tensors
    # tensor sets of onnx graph

    ## @var _symtab
    # symbolic table of IR node

    ## @var _input_names
    # input names of model


    ## The constructure
    def __init__(self, md, path):
        super().__init__(md)

        onnx_model = onnx.load(path)
        inferred_model = shape_inference.infer_shapes(onnx_model)
        # self.graph = gs.import_onnx(onnx_model)
        self.graph = gs.import_onnx(inferred_model)
        self.tensors = self.graph.tensors() #orderedDict of tensors
        self._symtab = self.__build_node_symtab(self.graph) #build name:node symtab
        self._input_names = [node.name for node in self.graph.inputs]

    ## This method is used to build symbolic table of IR node.
    #
    # @param self this object
    # @param onnx_graph graph of onnx model
    # @return symbolic table
    def __build_node_symtab(self, onnx_graph):
        symtab = dict()
        for node in onnx_graph.nodes:
            symtab[node.name] = node

        return symtab
    
    ## This method is used to emit the AIXTensor to be the input, output, scale, filter, biases, and variance.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        input_nodes = []
        input_tensors = []
        aix_tensors = []
        aix_tensor = None

        for pred in ir_node.preds:
            if pred.op != "Const" or pred.name in self._input_names:
                input_nodes.append(pred)

        for pred_node in input_nodes:
            if pred_node.name in self.tensors:
                input_tensors.append(self.tensors[pred_node.name])
            else:
                key = pred_node.output_name[0]
                input_tensors.append(self.tensors[key])

        for input_tensor in input_tensors:
            aix_tensor = AIXLayer.AIXTensor()
            dtype = input_tensor.dtype

            if not dtype:
                dtype = DEFAULT_DTYPE

            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)

            aix_tensor.dtype = aix_data_type_tbl.get(dtype.name)

            if dtype is None:
                data_format = DEFAULT_TYPE.encode()
            elif len(input_tensor.shape) == 1:
                data_format = b'VECTOR'
            else:
                data_format = DEFAULT_TYPE.encode()
            
            aix_tensor.format = aix_tensor_format_tbl[data_format]

            # If tensor is a constant, then the fval need to be set
            if type(input_tensor) is gs.ir.tensor.Constant:
                tensor_values = input_tensor.values.flatten()
                for fval in tensor_values:
                    aix_tensor.fval.append(fval)

            # Set aix tensor dims
            if input_tensor.shape:
                shape = list(map(lambda x: 1 if not x else x, input_tensor.shape))

                #handle unspecified "N" shape for input layer
                shape = list(map(lambda x: -1 if x == "N" else x, shape))

                shape.reverse() # AIXGraph shape is using NCHW format, in reversed order
                aix_tensor.dims.extend(shape)

            aix_tensor.size = int(np.prod(aix_tensor.dims))
            aix_tensors.append(aix_tensor)

        return aix_tensor
    

    ## This method is used to emit an AIX Tensors of an output type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        aix_tensor = self.__emit_aix_tensor(ir_node)
        return aix_tensor


    ## This method is used to emit an AIX Tensors to be the input, output, scale, filter, biase, and variance.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @is_inout_tensor in/output tensor checker
    # @return aix_tensor AIXTensor object
    def __emit_aix_tensor(self, ir_node, is_inout_tensor=False, **kwargs) -> AIXLayer.AIXTensor:
        aix_tensor = AIXLayer.AIXTensor()
        
        if ir_node.name in self.tensors:
            tensor = self.tensors[ir_node.name]
        else:
            tensor = self.tensors[ir_node.output_name[0]]

        dtype = tensor.dtype

        if not dtype:
            dtype = DEFAULT_DTYPE
        
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)

        aix_tensor.dtype = aix_data_type_tbl.get(dtype.name)

        #onnx data format is default at NCHW if any dtype is present then it is a vector
        if dtype is None:
            data_format = DEFAULT_TYPE.encode()
        elif len(tensor.shape) == 1:
            data_format = b'VECTOR'
        else:
            data_format = DEFAULT_TYPE.encode()
        
        # Set format
        aix_tensor.format = aix_tensor_format_tbl[data_format]

        # If tensor is a constant, then the fval need to be set
        if type(tensor) is gs.ir.tensor.Constant:
            tensor_values = tensor.values.flatten()
            for fval in tensor_values:
                aix_tensor.fval.append(fval)

        # Set aix tensor dims
        if tensor.shape:
            shape = list(map(lambda x: 1 if not x else x, tensor.shape))
            
            #handle unspecified "N" shape
            shape = list(map(lambda x: -1 if x == "N" else x, shape))
            
            #aixgraph shape is NCHW, in reversed order
            shape.reverse()
            aix_tensor.dims.extend(shape)

        else:
            logging.warning(f"AxfcONNXIRTranslator: {ir_node.name} shape is invalid.")

        aix_tensor.size = int(np.prod(aix_tensor.dims))

        return aix_tensor
    

    ###################################################################
    # protected methods
    ###################################################################

    ## This method emits some onnx-specific information of the given IR node
    # into the given AIX convolution layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX convolution layer
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer
        tensor = self.tensors[ir_node.output_name[0]]
        onnx_node = self._symtab[ir_node.name]

        #handling bias and weight in Conv
        for index, input in enumerate(onnx_node.inputs):
            if index == 0:
                continue #skip data variable
            
            for pred in ir_node.preds:
                if pred.name == input.name:
                    if len(input.shape) == 4: #weight, filter:
                        aix_layer.filter.CopyFrom(self.__emit_aix_tensor(pred))

                    if len(input.shape) == 1: #bias is optional in conv:
                        aix_layer.bias.CopyFrom(self.__emit_aix_tensor(pred))

        # Handle convolution description
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # Handle epsilon if present
        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']
        
        #GroupConv layer has to be handled the same in Conv since
        #Onnx doesn't have separate GroupConv (such as depthwise) Layer
        if aix_layer.convdesc.groups > 1:
            
            #DEBUG
            # print(f"{aix_layer.name}, group_conv: {aix_layer.convdesc.groups}")
            
            #change AIX layer type
            aix_layer.type.remove(AIXLayer.AIX_LAYER_CONVOLUTION)
            aix_layer.type.append(AIXLayer.AIX_LAYER_GROUP_CONV)

            #update filter size
            aix_layer.filter.size = int(aix_layer.filter.size / aix_layer.convdesc.groups)

        return AxfcError.SUCCESS

    ## This method emits an AIX tensor of bias type from the provided IR node.
    #
    # @param self The instance of the class.
    # @param ir_node The IR node to be converted into an AIX tensor.
    # @param onnx_node_attrs Additional attributes related to the ONNX node.
    # @return An AIX tensor of bias type.
    def __get_aix_tensor_bias(self, ir_node: AxfcIRNode, onnx_node_attrs: dict) -> AIXLayer.AIXTensor:
        node_inputs = ir_node.preds

        # Check if bias exists in attributes
        if "bias" in onnx_node_attrs:
            logging.info("Bias found in attributes, using the provided value.")
            return self.__emit_aix_tensor(onnx_node_attrs['bias'])

        # If no bias in attrs, check the node inputs (preds)
        for input in node_inputs:
            if 'bias' in input.name:
                logging.info("Bias found in preds, using the corresponding tensor.")
                return self.__emit_aix_tensor(input)

        # Set default bias
        return self.__emit_default_hyper_parameter(ir_node.aix_layer, 0)

    ## This method emits an AIX tensor of an filter type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an filter type
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, filter_node = None, **kwargs) -> AIXLayer.AIXTensor:
        
        
        if filter_node:
            aix_tensor = self.__emit_aix_tensor(filter_node)
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
        
        else:    
            node_inputs = ir_node.preds
            aix_tensor = None
            for input in node_inputs:
                if 'weight' in input.name:
                    aix_tensor = self.__emit_aix_tensor(input)
                    aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
        
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

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX batchnorm layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer
        
        #the order of the input layer is fixed and defined by onnx (input, gemma, beta, mean, variance)
        #ONNX Docs: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
        aix_layer.scale.CopyFrom(self.__emit_aix_tensor(ir_node.preds[1])) #gemma
        aix_layer.bias.CopyFrom(self.__emit_aix_tensor(ir_node.preds[2])) #beta
        aix_layer.mean.CopyFrom(self.__emit_aix_tensor(ir_node.preds[3])) #mean
        aix_layer.variance.CopyFrom(self.__emit_aix_tensor(ir_node.preds[4])) #variance

        #set the layer description, attributes.
        onnx_node = self._symtab[ir_node.name]
        for key, value in onnx_node.attrs.items():
            try:
                setattr(aix_layer, key, value)
            except Exception as e:
                logging.info(f"AxfcONNXIRTranslator:_emit_aix_layer_batchnorm {e}")

        return AxfcError.SUCCESS
    
    ##  This method emits some onnx-specific information of the given IR node
    # into the given AIX activation layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX activation layer
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_activation - node %d, %s",
                     ir_node.layer_id, ir_node.op)
        
        aix_layer = ir_node.aix_layer
        
        tensor = self.tensors.get(ir_node.name, None)
        if tensor:

            # filter
            aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

            # # convolution desc
            aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

            # if 'epsilon' in onnx_node.attrs:
            #     aix_layer.epsilon = onnx_node.attrs['epsilon']
        

        return AxfcError.SUCCESS

    ##  This method emits some onnx-specific information of the given IR node
    # into the given AIX maxpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX maxpool layer
    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.output_name[0]]
        onnx_node = self._symtab[ir_node.name]

        # filter 
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        # aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']

        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)
        aix_layer.samplingdesc.CopyFrom(sampling_desc)
        
        return AxfcError.SUCCESS
    
    
    ##  This method emits some onnx-specific information of the given IR node
    # into the given AIX avgpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.output_name[0]]
        onnx_node = self._symtab[ir_node.name]

        # epsilon
        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs.get('epsilon')
        
        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)

        # aixgraph requirement : set stride to default
        sampling_desc.stride[:] = []
        sampling_desc.stride.extend([0, 0, 0, 0])
        # aix_layer.filter.dims[0], aix_layer.filter.dims[1] = 1, 1

        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS

    ##  This method emits the AIX sampling description of the given IR node.
    #  This method must called after _emit_aix_convolution_desc()
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX sampling description
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXSamplingDesc:

        aix_layer = ir_node.aix_layer

        onnx_node = self._symtab[ir_node.name]

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcONNXIRTranslator:_emit_aix_sampling_desc - need TensorProto object')
        
        sampling_desc = AIXLayer.AIXSamplingDesc()

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

        #Attributes are fixed and followed in ONNX documentaiton:
        #https://onnx.ai/onnx/operators/onnx__MaxPool.html
        #window 
        if "kernel_shape" in onnx_node.attrs:
            stride_dict = dict(zip("HW", onnx_node.attrs.get("kernel_shape")))
            sampling_desc.window.extend([stride_dict['H'], stride_dict['W'], 0, 0])

            aix_layer.filter.dims[0] = stride_dict['H']
            aix_layer.filter.dims[1] = stride_dict['W']
        else:
            sampling_desc.window.extend([0,0,0,0])

        # strides
        if "strides" in onnx_node.attrs:
            stride_dict = dict(zip("HW", onnx_node.attrs.get("strides")))
            sampling_desc.stride.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            sampling_desc.stride.extend([1, 1, 0, 0])
        
        # padding
        if "pads" in onnx_node.attrs:
            sampling_desc.padding.extend(onnx_node.attrs.get("pads"))
        else:
            sampling_desc.padding.extend([0, 0, 0, 0])
        
        return sampling_desc
    
    ##  This method emits some onnx-specific information of the given IR node
    # into the given AIX element-wise add (ewadd) layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcONNXIRTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.tensors[ir_node.output_name[0]]
        onnx_node = self._symtab[ir_node.name]

        ewadddesc = AIXLayer.AIXEWAddDesc()
        scale_size = len(onnx_node.inputs)

        ewadddesc.scale.extend([1] * scale_size)

        aix_layer.ewadddesc.CopyFrom(ewadddesc)

        if 'epsilon' in onnx_node.attrs:
            aix_layer.epsilon = onnx_node.attrs['epsilon']
        
        return AxfcError.SUCCESS
    

    ##  This method emits the AIX convolution description of the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX convolution description
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
            convolution_desc.dilation.extend([0, 0, 0, 0])
        
        # group
        if 'group' in onnx_node.attrs:
            convolution_desc.groups = onnx_node.attrs['group']
        else:
            convolution_desc.groups = 1
        
        return convolution_desc