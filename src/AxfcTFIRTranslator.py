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
import math
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

    ###################################################################
    # private methods
    ###################################################################

    ## This method returns the data type of the given node_def
    #
    # @param tf_node_def input node_def
    # @return error info.
    def __get_aix_data_type(self, tf_node_def: tf.compat.v1.NodeDef) -> AIXLayer.AIXDataType:

        node_attr = tf_node_def.attr

        # get the data format string
        if "dtype" in node_attr:
            data_type = node_attr["dtype"].type
        elif "T" in node_attr:
            data_type = node_attr["T"].type
        else:
            return None

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

        node_attr = tf_node_def.attr

        # get the data format string
        if "data_format" in node_attr:
            data_format = node_attr["data_format"].s
        else:
            return None

        # return AIX tensor format
        try:
            return aix_tensor_format_tbl[data_format]
        except KeyError as e:
            logging.warning(e)

        return None

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

        # TODO: config when the aix_tensor_format is 'AIX_FORMAT_VECTOR'

        # query string data format from aix_tensor_format_tbl
        data_format = ([k for k, v in aix_tensor_format_tbl.items()
                        if v == tensor_format])[0].decode()

        # map the data format with its value
        return dict(zip(data_format, values))

    ###################################################################
    # protected methods
    ###################################################################

    ##  This method emits some tensorflow-specific information of the given IR node
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

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(input_nodes[1])
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # strides
        node_attr = tf_node_def.attr

        if "strides" in node_attr:
            strides = node_attr["strides"].list.i
        else:
            return AxfcError.INVALID_CONVOLUTION_LAYER

        # padding
        if "padding" in node_attr:
            padding = node_attr["padding"].s
        else:
            return AxfcError.INVALID_CONVOLUTION_LAYER

        # output - the current IR node

        # get data as dictionary has a key as the element of format
        input_dims = self.__get_aix_tensor_dims(input_tensor)
        filter_dims = self.__get_aix_tensor_dims(filter_tensor)
        strides = self.__get_values_of_format(strides, filter_tensor.format)

        if padding == b"SAME":
            output_h = input_dims['H']
            output_w = input_dims['W']
        elif padding == b"VALID":
            output_h = input_dims['H'] - filter_dims['H'] + 1
            output_w = input_dims['W'] - filter_dims['W'] + 1
        else:
            output_h = 0
            output_w = 0

        output_dims: list = [
            input_dims['N'],  # n
            math.ceil(output_h / strides['H']), # h
            math.ceil(output_w / strides['W']), # w
            filter_dims['C']  # c
        ]

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # bias - update using the calibration data
        bias_tensor = self._emit_aix_tensor_bias(ir_node, True)
        aix_layer.bias.CopyFrom(bias_tensor)

        # scale
        scale_tensor = self._emit_aix_tensor_scale(ir_node, True)
        aix_layer.scale.CopyFrom(scale_tensor)

        # mean
        mean_tensor = self._emit_aix_tensor_mean(ir_node, True)
        aix_layer.mean.CopyFrom(mean_tensor)

        # variance
        variance_tensor = self._emit_aix_tensor_variance(ir_node, True)
        aix_layer.variance.CopyFrom(variance_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # input_threshold - update using the calibration data

        # CHKME - YOUNGSUN (2020.08.10)
        # output_threshold - update using the calibration data

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
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
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(input_nodes[1])
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # strides
        node_attr = tf_node_def.attr

        if "strides" in node_attr:
            strides = node_attr["strides"].list.i
        else:
            return AxfcError.INVALID_CONVOLUTION_LAYER

        # padding
        if "padding" in node_attr:
            padding = node_attr["padding"].s
        else:
            return AxfcError.INVALID_CONVOLUTION_LAYER

        # output - the current IR node
        if padding == b"SAME":
            output_h = input_tensor.dims[1]
            output_w = input_tensor.dims[2]
        elif padding == b"VALID":
            output_h = input_tensor.dims[1] - filter_tensor.dims[0] + 1
            output_w = input_tensor.dims[2] - filter_tensor.dims[1] + 1
        else:
            output_h = 0
            output_w = 0

        output_dims: list = [
            input_tensor.dims[0], # n
            math.ceil(output_h / strides[1]), # h
            math.ceil(output_w / strides[2]), # w
            input_tensor.dims[3] * filter_tensor.dims[3] # c = k * channel_multiplier
        ]

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # bias - update using the calibration data
        bias_tensor = self._emit_aix_tensor_bias(ir_node, True)
        aix_layer.bias.CopyFrom(bias_tensor)

        # scale
        scale_tensor = self._emit_aix_tensor_scale(ir_node, True)
        aix_layer.scale.CopyFrom(scale_tensor)

        # mean
        mean_tensor = self._emit_aix_tensor_mean(ir_node, True)
        aix_layer.mean.CopyFrom(mean_tensor)

        # variance
        variance_tensor = self._emit_aix_tensor_variance(ir_node, True)
        aix_layer.variance.CopyFrom(variance_tensor)

        # CHKME - YOUNGSUN (2020.08.10)
        # input_threshold - update using the calibration data

        # CHKME - YOUNGSUN (2020.08.10)
        # output_threshold - update using the calibration data

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
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
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

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

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(ir_node, True)
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

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

    ##  This method emits some tensorflow-specific information of the given IR node
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
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

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

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(ir_node, True)
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # output
        output_dims: list = [
            1, # n
            1, # h
            1, # w
            input_tensor.dims[3] # c
        ]

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)
        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX maxpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        """
        tf.nn.max_pool(
            input, ksize, strides, padding, data_format=None, name=None
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

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

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(ir_node, True)
        filter_tensor.format = aix_tensor_format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # output
        node_attr = tf_node_def.attr

        if "_output_shapes" in node_attr:
            output_dims = list()

            for dim in node_attr["_output_shapes"].list.shape[0].dim:
                if dim.size < 0:
                    dim.size = 1
                output_dims.append(dim.size)
        else:
            output_dims: list = [
                1, # n
                1, # h
                1, # w
                input_tensor.dims[3] # c
            ]

        output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
        output_tensor.format = aix_tensor_format
        output_tensor.dtype = aix_data_type

        aix_layer.output.CopyFrom(output_tensor)

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)
        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX element-wise add (ewadd) layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        """
        tf.math.add(
            x, y, name=None
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

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
        filter_tensor = self._emit_aix_tensor_filter(ir_node, True)
        filter_tensor.format = input_tensor.format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # output
        node_attr = tf_node_def.attr

        if "_output_shapes" in node_attr:
            output_dims = list()

            for dim in node_attr["_output_shapes"].list.shape[0].dim:
                if dim.size < 0:
                    dim.size = 1
                output_dims.append(dim.size)

            output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
            output_tensor.format = input_tensor.format
            output_tensor.dtype = aix_data_type

            aix_layer.output.CopyFrom(output_tensor)
        else:
            aix_layer.output.CopyFrom(input_tensor)

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)
        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX softmax layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX softmax layer
    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_softmax - node %d", ir_node.layer_id)

        """
        tf.nn.softmax(
            logits, axis=None, name=None
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_tensor_format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.output

        aix_layer.input.CopyFrom(input_tensor)

        # output
        node_attr = tf_node_def.attr

        if "_output_shapes" in node_attr:
            output_dims = list()

            for dim in node_attr["_output_shapes"].list.shape[0].dim:
                if dim.size < 0:
                    dim.size = 1
                output_dims.append(dim.size)

            output_tensor = self._emit_aix_tensor_output(ir_node, output_dims)
            output_tensor.format = input_tensor.format
            output_tensor.dtype = aix_data_type

            aix_layer.output.CopyFrom(output_tensor)
        else:
            aix_layer.output.CopyFrom(input_tensor)

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)
        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
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
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

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

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_tensor_format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.output

        # output
        aix_layer.output.CopyFrom(input_tensor)

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX activation layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX activation layer
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_activation - node %d, %s",
                     ir_node.layer_id, ir_node.op)
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
        aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_SKIP_CONV)

        # tensorflow node_def for the given ir_node
        tf_node_def = ir_node.node_def

        # data type
        aix_data_type = self.__get_aix_data_type(tf_node_def)
        if aix_data_type is None:
            return AxfcError.INVALID_AIX_LAYER_TYPE

        # inputs/input
        err, input_nodes = self._get_emitted_input_nodes(ir_node)
        if err != AxfcError.SUCCESS:
            return err

        input_aix_layer = input_nodes[0].aix_layer

        if input_aix_layer is None:
            input_tensor = self._emit_aix_tensor_input(input_nodes[0])
            input_tensor.format = aix_layer.format
            input_tensor.dtype = aix_data_type
            input_tensor.ptr = 0
        else:
            input_tensor = input_aix_layer.output

        aix_layer.input.CopyFrom(input_tensor)

        # inputs/filter
        filter_tensor = self._emit_aix_tensor_filter(ir_node, True)
        filter_tensor.format = input_tensor.format
        filter_tensor.dtype = aix_data_type

        aix_layer.filter.CopyFrom(filter_tensor)

        # output
        aix_layer.output.CopyFrom(input_tensor)

        # convdesc
        convolution_desc = self._emit_aix_convolution_desc(ir_node)
        aix_layer.convdesc.CopyFrom(convolution_desc)

        return AxfcError.SUCCESS

    ##  This method emits an AIX tensor of an input type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX tensor of an input type
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        #logging.info("AxfcTFIRTranslator:_emit_aix_tensor_input - node %s", ir_node.name)

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
        
        dims: [batch_size, input_height, input_width, input_depth]
        """

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # get the Tensorflow node_def of the given node
        node_attr = ir_node.node_def.attr

        # tensor shape and size
        if "shape" in node_attr:                # Placeholder
            shape_dims = node_attr["shape"].shape.dim
        elif "_output_shapes" in node_attr:     # Pad
            shape_dims = node_attr["_output_shapes"].list.shape[0].dim
        else:
            logging.warning("_emit_aix_tensor_input: invalid shape - %s", str(node_attr))
            return None

        tensor_size = 1

        for dim in shape_dims:
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
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an filter type
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, is_default: bool = False) \
            -> AIXLayer.AIXTensor:
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

        dims: [filter_height, filter_width, filter_depth, number_of_filters]
        """

        # get the aix layer of thid node
        aix_layer = ir_node.aix_layer

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # configure the tensor with default scale values
        if is_default:
            input_dim = aix_layer.input.dims[3]

            aix_tensor.dims.append(1)
            aix_tensor.dims.append(input_dim)
            aix_tensor.dims.append(input_dim)
            aix_tensor.dims.append(1)

            return aix_tensor

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
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an bias type
    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, is_default: bool = False) \
            -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # CHKME - YOUNGSUN (2020.08.12)
        # get the bias (offset) values from the batchnorm node that follows this node.
        succ_node = ir_node.succs[0]

        if succ_node.op != "BatchNorm" and succ_node.op != "FusedBatchNorm":
            logging.warning("_emit_aix_tensor_bias: the successor is not a batchnorm node")
            return aix_tensor

        # get the offset (beta) node of the following batchnorm node
        input_nodes = list()

        for input_name in succ_node.node_def.input:
            input_nodes.append(self._ir_symtab[input_name])

        offset_node_def = input_nodes[2].node_def

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(offset_node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # get attribute values of the tensor node
        attr_value = offset_node_def.attr["value"]

        # tensor shape and size
        tensor_shape = attr_value.tensor.tensor_shape
        aix_tensor.size = tensor_shape.dim[0].size
        aix_tensor.dims.append(aix_tensor.size)

        # tensor_content
        for value in tf.make_ndarray(attr_value.tensor).flatten():
            aix_tensor.fval.append(value)

        return aix_tensor

    ##  This method emits an AIX tensor of an scale type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an scale type
    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, is_default: bool = False) \
            -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # configure the tensor with default scale values
        if is_default:
            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(1)

            return aix_tensor

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
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an mean type
    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, is_default: bool = False) \
            -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # configure the tensor with default scale values
        if is_default:
            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(0)

            return aix_tensor

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
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an variance type
    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, is_default: bool = False) \
            -> AIXLayer.AIXTensor:

        # create a new tensor
        aix_tensor = AIXLayer.AIXTensor()

        # dtype
        aix_tensor.dtype = self.__get_aix_data_type(ir_node.node_def)

        # tensor format
        aix_tensor.format = AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR

        # configure the tensor with default scale values
        if is_default:
            # dim (k)
            aix_tensor.size = ir_node.aix_layer.output.dims[3]
            aix_tensor.dims.append(aix_tensor.size)

            # fval
            for i in range(aix_tensor.size):
                aix_tensor.fval.append(1)

            return aix_tensor

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

        # get the aix layer and node_def of the given IR node
        aix_layer = ir_node.aix_layer
        tf_node_def = ir_node.node_def

        # get the layer info of the given IR node
        aix_layer_info = self._md.get_layer_info(ir_node.op)

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        # stride
        node_attr = tf_node_def.attr

        if "strides" in node_attr:
            strides = node_attr["strides"].list.i
        else:
            strides = [1, 1, 1, 1]

        for val in strides:
            convolution_desc.stride.append(val)

        # dilation
        if "dilation" in node_attr:
            dilations = node_attr["dilation"].list.i
        else:
            dilations = [1, 1, 1, 1]

        for val in dilations:
            convolution_desc.dilation.append(val)

        # padding
        # CHKME - YOUNGSUN (2020.08.10)
        # We need to check how to configure the values of padding.

        if "padding" in node_attr:
            padding = node_attr["padding"].s
        else:
            padding = None

        if padding is None:
            paddings = [0, 0, 0, 0]
        elif padding == b"VALID":
            paddings = [0, 0, 0, 0]
        elif padding == b"SAME":
            """
            if H1%Sh==0:
                padding along height=Ph=max(Fh−Sh,0)
            else:                
                padding along height=Ph=max(Fh−(H1%Sh),0)
                
            if W1%Sw==0:
                padding along width=Pw=max(Fw−Sw,0)
            else:
                padding along width=Pw=max(Fw−(W1%Sw),0)
            """

            # get data as dictionary has a key as element of format
            input_dims = self.__get_aix_tensor_dims(aix_layer.input)
            filter_dims = self.__get_aix_tensor_dims(aix_layer.filter)
            strides = self.__get_values_of_format(strides, aix_layer.filter.format)

            input_h = input_dims['H'] #aix_layer.input.dims[1]
            input_w = input_dims['W'] #aix_layer.input.dims[2]

            filter_h = filter_dims['H'] #aix_layer.filter.dims[0]
            filter_w = filter_dims['W'] #aix_layer.filter.dims[1]

            stride_h = strides['H'] #strides[1]
            stride_w = strides['W'] #strides[2]

            # for padding along for height
            if input_h % stride_h == 0:
                padding_h = max(0, (filter_h - stride_h))
            else:
                padding_h = max(0, (filter_h - (input_h % stride_h)))

            # for padding along for width
            if input_w % stride_w == 0:
                padding_w = max(0, (filter_w - stride_w))
            else:
                padding_w = max(0, (filter_w - (input_w % stride_w)))

            # padding top and left
            padding_t = math.floor(padding_h / 2)
            padding_l = math.floor(padding_w / 2)

            paddings = [
                padding_t,  # top
                padding_h - padding_t,  # bottom
                padding_l,  # left
                padding_w - padding_l  # right
            ]
        else:
            paddings = [0, 0, 0, 0]

        for val in paddings:
            convolution_desc.padding.append(val)

        # groups
        if not aix_layer_info.is_group:
            convolution_desc.groups = 1
        else:
            convolution_desc.groups = aix_layer.output.dims[3]

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
