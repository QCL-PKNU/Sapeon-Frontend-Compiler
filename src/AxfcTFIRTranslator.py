#######################################################################
#   AxfcTFIRTranslator
#
#   Created: 2020. 08. 07
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import logging
import numpy as np
import tensorflow as tf

from aixh_pb2 import *
from AxfcError import *
from AxfcIRNode import *
from AxfcIRBlock import *
from AxfcIRGraph import *
from AxfcMachineDesc import *
from AxfcIRTranslator import *

#######################################################################
# Global tables for AIXDataType and AIXTensorFormat
#######################################################################

# AIXDataType table
aix_data_type_tbl = {
    tf.float16: AIXLayer.AIXDataType.AIX_DATA_HALF,
    tf.float32: AIXLayer.AIXDataType.AIX_DATA_FLOAT,
    tf.float64: AIXLayer.AIXDataType.AIX_DATA_DOUBLE,
    tf.uint8: AIXLayer.AIXDataType.AIX_DATA_UINT8,
    tf.int8: AIXLayer.AIXDataType.AIX_DATA_SINT8,
    tf.int16: AIXLayer.AIXDataType.AIX_DATA_SINT16
}

# AIXTensorFormat table
aix_tensor_format_tbl = {
    b"NCHW": AIXLayer.AIXTensorFormat.AIX_FORMAT_NCHW,
    b"NHWC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NHWC,
    b"NWHC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NWHC,
    b"VECTOR": AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR
}

#######################################################################
# AxfcTFIRTranslator class
#######################################################################

class AxfcTFIRTranslator(AxfcIRTranslator):

    ## @var __aix_data_type_tbl
    # a table for AIXDataType

    ## @var __aix_tensor_format_tbl
    # a table for AIXTensorFormat

    ## The constructor
    def __init__(self, md):
        super().__init__(md)

    ## This method returns the data type of the given node_def
    #
    # @param tf_node_def input node_def
    # @return error info.
    def __get_aix_data_type(self, tf_node_def: tf.compat.v1.NodeDef) -> AIXLayer.AIXDataType:

        # get the data format string
        if tf_node_def.op == "Const":
            data_type = tf_node_def.attr["dtype"].type
        else:
            data_type = tf_node_def.attr["T"].type

        # return AIX data type
        try:
            return aix_data_type_tbl[data_type]
        except KeyError as e:
            logging.warning(e)

        return None

    ## This method returns the tensor format of the given node_def
    #
    # @param tf_node_def input node_def
    # @return error info.
    def __get_aix_tensor_format(self, tf_node_def: tf.compat.v1.NodeDef) -> AIXLayer.AIXTensorFormat:

        try:
            # get the data format string
            data_format = tf_node_def.attr["data_format"].s

            # return AIX tensor format
            return aix_tensor_format_tbl[data_format]
        except KeyError as e:
            logging.warning(e)

        return None

    ##  This method emits some convolution-specific information of the given IR node
    # into the given AIX convolution layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX convolution layer
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        """
        tf.nn.conv2d(
            input, filters, strides, padding, data_format, dilations, name
        )
        """

        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # tensor format
        aix_tensor_format = self.__get_aix_tensor_format(tf_node_def)
        if aix_tensor_format is None:
            return AxfcError.INVALID_AIX_TENSOR_FORMAT

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # inputs/input
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_tensor_format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.input

        aix_layer.input.CopyFrom(input_tensor)

        # inputs/filter
        filter_tensor = self._emit_aix_tensor_filter(input_nodes[1])
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # strides
        strides = tf_node_def.attr["strides"].list.i

        # output - the current IR node
        try:
            output_dims: list = [
                input_tensor.dims[0], # b
                input_tensor.dims[1] // strides[1], # i
                input_tensor.dims[2] // strides[2], # j
                filter_tensor.dims[3] # k
            ]
        except IndexError as e:
            logging.warning("_emit_aix_layer_convolution: %s", e)
            return AxfcError.INVALID_AIX_TENSOR_INPUT

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # bias - update using the calibration data
        bias_tensor = self._emit_aix_tensor_bias(ir_node)
        aix_layer.bias.CopyFrom(bias_tensor)

        # scale
        scale_tensor = self._emit_aix_tensor_scale(ir_node)
        aix_layer.scale.CopyFrom(scale_tensor)

        # mean
        mean_tensor = self._emit_aix_tensor_mean(ir_node)
        aix_layer.scale.CopyFrom(mean_tensor)

        # variance
        variance_tensor = self._emit_aix_tensor_variance(ir_node)
        aix_layer.variance.CopyFrom(variance_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # input_threshold - update using the calibration data

        # CHKME - YOUNGSUN (2020.08.10)
        # output_threshold - update using the calibration data

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)

        return AxfcError.SUCCESS

    ##  This method emits some group convolution-specific information of the given IR node
    # into the given AIX group convolution layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX convolution layer
    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        """
        tf.nn.conv2d(
            input, filters, strides, padding, data_format, dilations, name
        )
        """

        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # tensor format
        aix_tensor_format = self.__get_aix_tensor_format(tf_node_def)
        if aix_tensor_format is None:
            return AxfcError.INVALID_AIX_TENSOR_FORMAT

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # type
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # inputs/input
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_tensor_format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.input

        aix_layer.input.CopyFrom(input_tensor)

        # inputs/filter
        filter_tensor = self._emit_aix_tensor_filter(input_nodes[1])
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # strides
        strides = tf_node_def.attr["strides"].list.i

        # output - the current IR node
        try:
            output_dims: list = [
                input_tensor.dims[0], # b
                input_tensor.dims[1] // strides[1], # i
                input_tensor.dims[2] // strides[2], # j
                filter_tensor.dims[3] # k
            ]
        except IndexError as e:
            logging.warning("_emit_aix_layer_convolution: %s", e)
            return AxfcError.INVALID_AIX_TENSOR_INPUT

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        # scale

        # bias

        # mean

        # variance

        # convdesc

        aix_layer.output.CopyFrom(output_tensor)

        return AxfcError.SUCCESS

    ##  This method emits some convolution-specific information of the given IR node
    # into the given AIX batchnorm layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        """
        tf.nn.batch_normalization(
            x, mean, variance, offset, scale, variance_epsilon, name=None
        )
        """

        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # tensor format
        aix_tensor_format = self.__get_aix_tensor_format(tf_node_def)
        if aix_tensor_format is None:
            return AxfcError.INVALID_AIX_TENSOR_FORMAT

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # type
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # inputs/input
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_tensor_format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.output

        aix_layer.input.CopyFrom(input_tensor)

        # inputs/filter

        # output - the current IR node
        output_tensor = self._emit_aix_tensor_output(ir_node, input_tensor.dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # scale
        scale_tensor = self._emit_aix_tensor_scale(input_nodes[1])
        aix_layer.scale.CopyFrom(scale_tensor)

        # mean
        mean_tensor = self._emit_aix_tensor_mean(input_nodes[3])
        aix_layer.mean.CopyFrom(mean_tensor)

        # variance
        variance_tensor = self._emit_aix_tensor_variance(input_nodes[4])
        aix_layer.variance.CopyFrom(variance_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # output_threshold - update using the calibration data

        # epsilon
        aix_layer.epsilon = tf_node_def.attr["epsilon"].f

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)

        return AxfcError.SUCCESS

    ##  This method emits some convolution-specific information of the given IR node
    # into the given AIX avgpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        """
        tf.nn.avg_pool(
            input, ksize, strides, padding, data_format, name
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # input layer
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            return AxfcError.INVALID_BATCHNORM_LAYER

        # type
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # value
        input_tensor = input_aix_layer.output
        aix_layer.input.CopyFrom(input_tensor)

        # output
        aix_layer.output.CopyFrom(input_tensor)

        return AxfcError.SUCCESS

    ##  This method emits some convolution-specific information of the given IR node
    # into the given AIX biasadd layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_biasadd - node %d", ir_node.layer_id)

        """
        tf.nn.bias_add(
            input, ksize, strides, padding, data_format, name
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # input layer
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            return AxfcError.INVALID_BATCHNORM_LAYER

        # type
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # value
        input_tensor = input_aix_layer.output
        aix_layer.input.CopyFrom(input_tensor)

        # output
        aix_layer.output.CopyFrom(input_tensor)

        return AxfcError.SUCCESS

    ##  This method emits some convolution-specific information of the given IR node
    # into the given AIX activation layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX activation layer
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_activation - node %d", ir_node.layer_id)

        """
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=None, logits=None, name=None
        )
        tf.nn.relu(
            features, name=None
        )
        tf.nn.relu6(
            features, name=None
        )
        tf.nn.leaky_relu(
            features, alpha=0.2, name=None
        )
        tf.keras.layers.PReLU(
            alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
            shared_axes=None, **kwargs
        )
        tf.math.tanh(
            x, name=None
        )
        """

        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # emit tensor inputs
        input_nodes = list()

        for input_name in tf_node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        # input layer
        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            return AxfcError.INVALID_ACTIVATION_LAYER

        # type
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # input
        input_tensor = input_aix_layer.output
        aix_layer.input.CopyFrom(input_tensor)

        # output
        aix_layer.output.CopyFrom(input_tensor)

        # filter

        return AxfcError.SUCCESS

    ##  This method emits an AIX tensor of an input type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an input type
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        logging.info("AxfcTFIRTranslator:_emit_aix_tensor_input - node %s", ir_node.name)

        """
        input {
            dtype: AIX_DATA_FLOAT
            format: AIX_FORMAT_NCHW
            dims: 224
            dims: 224
            dims: 3
            dims: 1
            size: 150528
            ptr: 94234087239120
        }
        """

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # get the Tensorflow node_def of the given node
        tf_node_def = ir_node.node_def

        # tensor shape and size
        attr_value = tf_node_def.attr["shape"]
        tensor_size = 1

        for dim in attr_value.shape.dim:
            if dim.size < 0:
                dim.size = 1
            aix_tensor.dims.append(dim.size)
            tensor_size *= dim.size

        aix_tensor.size = tensor_size

        return aix_tensor

    ##  This method emits an AIX tensor of an filter type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an filter type
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:

        """
        filter {
            dtype: AIX_DATA_FLOAT
            format: AIX_FORMAT_NCHW
            dims: 3
            dims: 3
            dims: 32
            dims: 32
            fval: -1.01131923e-06
            fval: -1.87250947e-07
            fval: 6.8427272e-07
            ...
        """

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # get the Tensorflow node_def of the given node
        tf_node_def = ir_node.node_def
        attr_value = tf_node_def.attr["value"]

        # tensor shape and size
        tensor_shape = attr_value.tensor.tensor_shape
        tensor_size = 1

        for dim in tensor_shape.dim:
            aix_tensor.dims.append(dim.size)
            tensor_size *= dim.size

        aix_tensor.size = tensor_size

        # tensor_content
        for value in tf.make_ndarray(attr_value.tensor).flatten():
            aix_tensor.fval.append(value)

        return aix_tensor

    ##  This method emits an AIX tensor of an bias type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an bias type
    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # dim (k)
        aix_tensor.size = ir_node.aix_layer.filter.dims[3]
        aix_tensor.dims.append(aix_tensor.size)

        # fval
        #for i in range(aix_tensor.size):
        #    aix_tensor.fval.append(1)

        return aix_tensor

    ##  This method emits an AIX tensor of an scale type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an scale type
    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # configure the tensor with default scale values
        if ir_node.op != "Const":

            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(1)

            return aix_tensor

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # get the Tensorflow node_def of the given node
        tf_node_def = ir_node.node_def
        attr_value = tf_node_def.attr["value"]

        # tensor shape and size
        tensor_shape = attr_value.tensor.tensor_shape
        aix_tensor.size = tensor_shape.dim[0].size
        aix_tensor.dims.append(aix_tensor.size)

        # tensor_content
        for value in tf.make_ndarray(attr_value.tensor).flatten():
            aix_tensor.fval.append(value)

        return aix_tensor

    ##  This method emits an AIX tensor of an mean type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an mean type
    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # configure the tensor with default scale values
        if ir_node.op != "Const":

            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(0)

            return aix_tensor

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # get the Tensorflow node_def of the given node
        tf_node_def = ir_node.node_def
        attr_value = tf_node_def.attr["value"]

        # tensor shape and size
        tensor_shape = attr_value.tensor.tensor_shape
        aix_tensor.size = tensor_shape.dim[0].size
        aix_tensor.dims.append(aix_tensor.size)

        # tensor_content
        for value in tf.make_ndarray(attr_value.tensor).flatten():
            aix_tensor.fval.append(value)

        return aix_tensor

    ##  This method emits an AIX tensor of an variance type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an variance type
    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # configure the tensor with default scale values
        if ir_node.op != "Const":

            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(1)

            return aix_tensor

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # get the Tensorflow node_def of the given node
        tf_node_def = ir_node.node_def
        attr_value = tf_node_def.attr["value"]

        # tensor shape and size
        tensor_shape = attr_value.tensor.tensor_shape
        aix_tensor.size = tensor_shape.dim[0].size
        aix_tensor.dims.append(aix_tensor.size)

        # tensor_content
        for value in tf.make_ndarray(attr_value.tensor).flatten():
            aix_tensor.fval.append(value)

        return aix_tensor

    ##  This method emits an AIX tensor of an output type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param output_dims output dimensions
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, output_dims: list) -> AIXLayer.AIXTensor:
        """
         output[b, i, j, k] =
            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                            filter[di, dj, q, k]
        """

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # strides
        output_size = 1

        for dim in output_dims:
            aix_tensor.dims.append(dim)

            # calculate the total output size
            output_size *= dim

        aix_tensor.size = output_size

        return aix_tensor

    ##  This method emits the AIX convolution description of the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX convolution description
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXConvolutionDesc:

        convolution_desc = AIXLayer.AIXConvolutionDesc()

        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        # get the aix layer info of the given IR node
        aix_layer_info = self._md.get_layer_info(ir_node.op)

        # dtype
        convolution_desc.dtype = self.__get_aix_data_type(ir_node.node_def)

        # padding

        # stride

        # dilation

        # groups
        if not aix_layer_info.is_group:
            convolution_desc.groups = 1
        else:
            convolution_desc.groups = aix_layer.filter.dims[3]

        return convolution_desc

    ##  This method emits the AIX sampling description of the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX sampling description
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXSamplingDesc:

        sampling_desc = AIXLayer.AIXSamplingDesc()

        # mode

        # padding

        # stride

        # window

        # groups

        return sampling_desc