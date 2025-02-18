#######################################################################
#   AxfcTFIRTranslator
#
#   Created: 2020. 08. 07
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################

import tensorflow as tf
import numpy as np

from .AxfcIRTranslator import *
from util import loadFrozenModel, print_tensor_content

#######################################################################
# Global tables for AIXDataType and AIXTensorFormat
#######################################################################

## AIXDataType table
aix_data_type_tbl = {
    tf.float16: AIXLayer.AIXDataType.AIX_DATA_HALF,
    tf.float32: AIXLayer.AIXDataType.AIX_DATA_FLOAT,
    tf.float64: AIXLayer.AIXDataType.AIX_DATA_DOUBLE,
    tf.uint8: AIXLayer.AIXDataType.AIX_DATA_UINT8,
    tf.int8: AIXLayer.AIXDataType.AIX_DATA_SINT8,
    tf.int16: AIXLayer.AIXDataType.AIX_DATA_SINT16
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

#######################################################################
# AxfcTFIRTranslator class
#######################################################################

class AxfcTFIRTranslator(AxfcIRTranslator):

    def __init__(self, md, path, **kwargs):
        super().__init__(md)

        graph_def = loadFrozenModel(path)
        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def, name='')


    def __emit_default_hyper_parameter(self, aix_layer: AIXLayer, default_value: int, 
                                       dims_key='C') -> AIXLayer.AIXTensor:
        """
        Emits default hyperparameters: scale, mean, variance.

        Args:
            aix_layer (AIXLayer): 
        """
        tensor = AIXLayer.AIXTensor()
        tensor.dtype = aix_layer.output.dtype
        tensor.format = AIXLayer.AIX_FORMAT_VECTOR
        output_channel = self.__get_tensor_dims(aix_layer.output)[dims_key]
        tensor.dims.append(output_channel)
        tensor.size = output_channel
        tensor.fval.extend([default_value] * output_channel)
        return tensor
    

    def __emit_tensor(self, tensor, is_inout_tensor=False, 
                      default_format=b'NCHW', default_value=1, **kwargs) -> AIXLayer.AIXTensor:

        aix_tensor = AIXLayer.AIXTensor()

        # Set dtype
        aix_tensor.dtype = aix_data_type_tbl.get(
            tensor.dtype, AIXLayer.AIXDataType.AIX_DATA_FLOAT
        )
        if aix_tensor.dtype == AIXLayer.AIXDataType.AIX_DATA_FLOAT:
            logging.warning(f"Unsupported tensor data type: {tensor.dtype}, defaulting to FLOAT.")

        # Determine data format
        if "data_format" in tensor.op.node_def.attr:
            data_format = tensor.op.node_def.attr["data_format"].s
        elif tensor.shape.ndims == 1:
            data_format = b'VECTOR'
        else:
            data_format = DEFAULT_TYPE.encode()

        aix_tensor.format = aix_tensor_format_tbl.get(data_format, aix_tensor_format_tbl[b'NCHW'])
        if aix_tensor.format == aix_tensor_format_tbl[b'NCHW']:
            logging.warning(f"Unsupported tensor format: {data_format}. Using default NCHW format.")

        # Populate fval
        dims = print_tensor_content(tensor.op)

        if dims is not None:
            if data_format != b'VECTOR':
                # Reshape the dimensions for NCHW or NHWC
                try:
                    if data_format == b'NHWC':
                        new_dims = np.einsum('HWIO->OIHW', dims)
                    else:
                        new_dims = dims  # For NCHW, keep as is
                    tensor_values = new_dims.flatten()
                except ValueError as e:
                    logging.error(f"Error reshaping tensor dimensions: {e}")
                    tensor_values = dims.flatten()

            else:
                tensor_values = dims.flatten()

            # Add tensor values to fval
            for dim in tensor_values:
                aix_tensor.fval.append(dim)

            # Log if fval is successfully populated
            if not aix_tensor.fval:
                logging.warning(f"fval not populated for tensor: {tensor.name}")


        # Set dims
        shape = list(map(lambda x: 1 if not x else x, tensor.shape))
        if is_inout_tensor:
            # Convert shape to NCHW format
            shape_dict = dict(zip('NHWC', shape))
            for dim in reversed('NCHW'):
                aix_tensor.dims.append(shape_dict.get(dim, 1))
        else:
            aix_tensor.dims.extend(shape)

        # Set size
        aix_tensor.size = np.prod(aix_tensor.dims) if aix_tensor.dims else 1

        return aix_tensor


    def __get_tensor_by_name(self, name: str):
        """Get tensor by name."""

        # check the BiasaddClone
        postfix_name = name.split('/')[-1]
        if postfix_name == 'BiasaddClone':
            name = name.replace('/BiasaddClone', '')

        return self.graph.get_tensor_by_name(name + ':0')
    

    def __get_tensor_dims(self, aix_tensor: AIXLayer.AIXTensor) -> dict:
        """
        Gets aix_tensor from format as dictionary.
        """
        return self.__get_values_of_format(aix_tensor.dims, aix_tensor.format)
    

    def __get_values_of_format(self, values: list,
                               tensor_format: AIXLayer.AIXTensorFormat) -> dict:
        """
        Gets data from aix_tensor format as dictionary.
        """

        # query string data format from aix_tensor_format_tbl
        data_format = ([k for k, v in aix_tensor_format_tbl.items()
                        if v == tensor_format])[0].decode()

        # Darknet dims tensor is reverse
        reverse_values = list(values)
        reverse_values.reverse()

        # map the data format with its value
        return dict(zip(data_format, reverse_values))
    

    ###################################################################
    # Emission Methods
    ###################################################################


    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIX tensor of an input type, given the IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_tensor_input - node %s", ir_node.name)

        # get TensorProto object
        tensor = self.__get_tensor_by_name(ir_node.name)

        # get input Tensor
        input_tensors = list(filter(lambda x: x.op.type != 'Const', tensor.op.inputs))
        if input_tensors:
            #Leanghok - only one input is being emitted !IMPORTANT
            logging.warning("AxfcTFIRTranslator.py: Only 1 input tensor information is being emitted to aix graph")
            aix_tensor = self.__emit_tensor(input_tensors[0], is_inout_tensor=True)

        return aix_tensor


    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIX tensor of an output type, given the IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        tensor = self.__get_tensor_by_name(ir_node.name)

        aix_tensor = (self.__emit_tensor(tensor, is_inout_tensor=True))

        return aix_tensor


    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIX tensor of a filter type from the given IR Node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        """
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_filter - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]
        aix_tensor = None

        for idx, tensor in enumerate(tensors):
            if idx == 1:
                aix_tensor = self.__emit_tensor(tensor)

                dims_dict = self.__get_tensor_dims(ir_node.aix_layer.output)

                # Darknet requirement
                aix_tensor.format = aix_tensor_format_tbl[b'NCHW']

                aix_tensor.dims[-1] = dims_dict['C']
                break
                

        # Handle default case if no matching tensor is found
        if aix_tensor is None:
            logging.warning("Filter tensor not found; setting default tensor values")
            aix_layer = ir_node.aix_layer
            aix_tensor = AIXLayer.AIXTensor()
            aix_tensor.dtype = aix_layer.input.dtype
            aix_tensor.format = aix_tensor_format_tbl[b'NCHW']
            aix_tensor.dims.append(1)  # width
            aix_tensor.dims.append(1)  # height

            # Set input and output channel dimensions
            input_dims_dict = self.__get_tensor_dims(aix_layer.input)
            output_dims_dict = self.__get_tensor_dims(aix_layer.output)
            aix_tensor.dims.append(input_dims_dict['C'])  # input channels
            aix_tensor.dims.append(output_dims_dict['C'])  # output channels

            # Debug log for default fval
            logging.warning("Default filter tensor created; fval is empty")

        return aix_tensor


    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIXTensor of a scale type from the given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            aix_tensor (AIXLayer.AIXTensor): AIX tensor of an scale type.
        """
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_scale - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'gamma' in tensor.name:
                aix_tensor = self.__emit_tensor(tensor)
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    

    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIXTensor of a bias type from the given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            aix_tensor (AIXLayer.AIXTensor): AIX tensor of a bias type.
        """
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_bias - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'biases' in tensor.name or 'beta' in tensor.name:
                aix_tensor = self.__emit_tensor(tensor)
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 0)

        return aix_tensor


    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIXTensor of a variance type from the given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            aix_tensor (AIXLayer.AIXTensor): AIX tensor of a variance type.
        """
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_mean - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'moving_variance' in tensor.name:
                aix_tensor = self.__emit_tensor(tensor, optimize=True)
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor
    

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
        Emits an AIXTensor of a mean type from the given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            aix_tensor (AIXLayer.AIXTensor): AIX tensor of a variance type.
        """

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_mean - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'moving_mean' in tensor.name:
                aix_tensor = self.__emit_tensor(tensor)
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 0)

        return aix_tensor
    

    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXConvolutionDesc:
        """
        Emits AIX convolution description of the given IR Node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            An AIX convolution description.
        """

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator: _emit_aix_convolution_desc - need TensorProto object')

        if 'is_default' in kwargs:
            is_default = kwargs['is_default']
        else:
            is_default = False

        convolution_desc = AIXLayer.AIXConvolutionDesc()

        aix_layer = ir_node.aix_layer

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        # strides
        if 'strides' in tensor.op.node_def.attr and not is_default:
            stride_dict = dict(zip('AHWB', tensor.op.get_attr('strides')))
            convolution_desc.stride.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            convolution_desc.stride.extend([1, 1, 0, 0])

        # paddings
        if 'padding' in tensor.op.node_def.attr and not is_default:

            if tensor.op.get_attr('padding') == b'VALID':
                convolution_desc.padding.extend([0, 0, 0, 0])
            else:  # SAME

                input_dims_dict = self.__get_tensor_dims(aix_layer.input)

                input_h = input_dims_dict['H']
                stride_h = convolution_desc.stride[0]
                filter_h = aix_layer.filter.dims[0]

                input_w = input_dims_dict['W']
                stride_w = convolution_desc.stride[1]
                filter_w = aix_layer.filter.dims[1]

                if input_h % stride_h == 0:
                    pad_along_height = max((filter_h - stride_h), 0)
                else:
                    pad_along_height = max(filter_h - (input_h % stride_h), 0)
                if input_w % stride_w == 0:
                    pad_along_width = max((filter_w - stride_w), 0)
                else:
                    pad_along_width = max(filter_w - (input_w % stride_w), 0)

                ## Tensorflow system
                # pad_top = pad_along_height // 2
                # pad_bottom = pad_along_height - pad_top
                # pad_left = pad_along_width // 2
                # pad_right = pad_along_width - pad_left

                ## Darknet system
                pad_bottom = pad_along_height // 2
                pad_top = pad_along_height - pad_bottom
                pad_right = pad_along_width // 2
                pad_left = pad_along_width - pad_right

                convolution_desc.padding.extend([pad_top, pad_bottom, pad_left, pad_right])

        else:
            convolution_desc.padding.extend([0, 0, 0, 0])

        # pad layer
        for input_tensor in tensor.op.inputs:
            if input_tensor.op.type == 'Pad' and not is_default:
                for pad in input_tensor.op.inputs:
                    if pad.op.type == 'Const':
                        value = print_tensor_content(pad.op)

                        pad_top = value[1][0]
                        pad_bottom = value[1][1]
                        pad_left = value[2][0]
                        pad_right = value[2][1]
                        convolution_desc.padding[:] = []
                        convolution_desc.padding.extend([pad_top, pad_bottom, pad_left, pad_right])

                        # config with width and height of input
                        input_format = aix_layer.input.format
                        data_format = ([k for k, v in aix_tensor_format_tbl.items() if v == input_format])[0].decode()
                        reverse_format = data_format[::-1]

                        index_width = reverse_format.find('W')
                        index_height = reverse_format.find('H')

                        # re-set width and height of input dims
                        aix_layer.input.dims[index_width] -= (pad_left + pad_right)
                        aix_layer.input.dims[index_height] -= (pad_top + pad_bottom)

                        # re-set input size
                        input_dims_dict = self.__get_tensor_dims(aix_layer.input)
                        aix_layer.input.size = input_dims_dict['W'] * input_dims_dict['H'] * input_dims_dict['C']

        # dilation
        if 'dilations' in tensor.op.node_def.attr and not is_default:
            convolution_desc.dilation.extend(tensor.op.get_attr('dilations'))
        else:
            convolution_desc.dilation.extend([1, 1, 1, 1])

        # groups
        if AIXLayer.AIX_LAYER_GROUP_CONV in aix_layer.type and not is_default:
            input_dims_dict = self.__get_tensor_dims(aix_layer.input)
            convolution_desc.groups = input_dims_dict['C']
        else:
            convolution_desc.groups = 1

        return convolution_desc


    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXSamplingDesc:
        """
        Emits the sampling description of the given IR node.
        This method is used in AIXLayer.AIX_LAYER_MAXPOOL,
                                 AIXLayer.AIX_LAYER_AVGPOOL,
                                 AIXLayer.AIX_LAYER_UPSAMPLE,
                                 AIXLayer.AIX_LAYER_REORG
        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        Returns:
            An AIX sampling description.
        """

        aix_layer = ir_node.aix_layer

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_sampling_desc - need TensorProto object')

        sampling_desc = AIXLayer.AIXSamplingDesc()

        # mode
        # layer_type = ir_node.aix_layer.type[-1]

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

        # window
        if 'ksize' in tensor.op.node_def.attr:
            stride_dict = dict(zip('AHWB', tensor.op.get_attr('ksize')))
            sampling_desc.window.extend([stride_dict['H'], stride_dict['W'], 0, 0])

            # TODO : check the relationship between window and stride in convc
            aix_layer.filter.dims[0] = stride_dict['H']
            aix_layer.filter.dims[1] = stride_dict['W']
        else:
            sampling_desc.window.extend([1, 1, 0, 0])

        # strides
        if 'strides' in tensor.op.node_def.attr:
            stride_dict = dict(zip('AHWB', tensor.op.get_attr('strides')))
            sampling_desc.stride.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            sampling_desc.stride.extend([1, 1, 0, 0])

        # padding
        if 'padding' in tensor.op.node_def.attr:

            if tensor.op.get_attr('padding') == b'VALID':
                sampling_desc.padding.extend([0, 0, 0, 0])
            else:  # SAME

                input_dims_dict = self.__get_tensor_dims(aix_layer.input)

                input_h = input_dims_dict['H']
                stride_h = sampling_desc.stride[0]
                filter_h = sampling_desc.window[0]

                input_w = input_dims_dict['W']
                stride_w = sampling_desc.stride[1]
                filter_w = sampling_desc.window[1]

                if input_h % stride_h == 0:
                    pad_along_height = max((filter_h - stride_h), 0)
                else:
                    pad_along_height = max(filter_h - (input_h % stride_h), 0)
                if input_w % stride_w == 0:
                    pad_along_width = max((filter_w - stride_w), 0)
                else:
                    pad_along_width = max(filter_w - (input_w % stride_w), 0)

                ## Tensorflow system
                # pad_top = pad_along_height // 2
                # pad_bottom = pad_along_height - pad_top
                # pad_left = pad_along_width // 2
                # pad_right = pad_along_width - pad_left

                ## Darknet system
                pad_bottom = pad_along_height // 2
                pad_top = pad_along_height - pad_bottom
                pad_right = pad_along_width // 2
                pad_left = pad_along_width - pad_right

                sampling_desc.padding.extend([pad_top, pad_bottom, pad_left, pad_right])

        else:
            sampling_desc.padding.extend([0, 0, 0, 0])

        return sampling_desc


    ###################################################################
    # Emission methods
    ###################################################################


    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of convolution layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter, scale, mean, variance
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.scale.CopyFrom(self._emit_aix_tensor_scale(ir_node, tensor=tensor))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node, tensor=tensor))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node, tensor=tensor))
        # aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        return AxfcError.SUCCESS


    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor information of group convolution layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.scale.CopyFrom(self._emit_aix_tensor_scale(ir_node, tensor=tensor))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node, tensor=tensor))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node, tensor=tensor))
        # aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        return AxfcError.SUCCESS


    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of batchnorm layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter, scale, mean, variance
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.scale.CopyFrom(self._emit_aix_tensor_scale(ir_node, tensor=tensor))
        aix_layer.mean.CopyFrom(self._emit_aix_tensor_mean(ir_node, tensor=tensor))
        aix_layer.variance.CopyFrom(self._emit_aix_tensor_variance(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        # bias
        if 'FusedBatchNorm' in ir_node.op:
            bias_tensor = self.__get_tensor_by_name(ir_node.name + '/BiasaddClone')
            aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=bias_tensor))

        return AxfcError.SUCCESS


    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of avgpool layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        filter = self._emit_aix_tensor_filter(ir_node, tensor=tensor)
        aix_layer.filter.CopyFrom(filter)

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor, is_default=True))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)

        # darknet : set stride to default
        sampling_desc.stride[:] = []
        sampling_desc.stride.extend([0, 0, 0, 0])
        aix_layer.filter.dims[0], aix_layer.filter.dims[1] = 1, 1

        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS


    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of maxpool layer, given IR node.
        tf.nn.max_pool(
            input, ksize, strides, padding, data_format=None, name=None
        )

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor, is_default=True))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)
        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS


    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of element-wise add (ewadd) layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            An output AIX element-wise add (ewadd) layer.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        # Get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # Get tensor
        tensor = self.__get_tensor_by_name(ir_node.name)

        # Emit tensor information
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # EWAddesc
        ewadddesc = AIXLayer.AIXEWAddDesc()
        scale_size = len(tensor.op.inputs)

        ewadddesc.scale.extend([1] * scale_size)

        aix_layer.ewadddesc.CopyFrom(ewadddesc)

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0
        return AxfcError.SUCCESS
    

    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of softmax layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            An output AIX softmax layer.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_softmax - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # Emit tensor
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        return AxfcError.SUCCESS


    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of biasadd layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.

        Returns:
            An output AIX biasadd layer.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_biasadd - node %d", ir_node.layer_id)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        #     tf.nn.bias_add(
        #    value, bias, data_format=None, name=None
        #     )
        #bias_add always have bias in the second input.
        #https://www.tensorflow.org/api_docs/python/tf/nn/bias_add
        # aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=tensor))
        bias_tensor = self.__get_tensor_by_name(ir_node.preds[1].name)
        aix_layer.bias.CopyFrom(self.__emit_tensor(bias_tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        return AxfcError.SUCCESS


    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        """
        Emits tensor inforamtion of activation layer, given IR node.

        Args:
            ir_node (AxfcIRNode): Object of an IRNode.
            **kwargs: Additional arguments.
        """
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_activation - node %d, %s",
                     ir_node.layer_id, ir_node.op)

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        # else:
        #     aix_layer.epsilon = 0

        return AxfcError.SUCCESS
