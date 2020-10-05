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

import math
import tensorflow as tf

from AxfcIRTranslator import *

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

# following the Darknet format
DEFAULT_TYPE = 'NCHW'

#######################################################################
# AxfcTFIRTranslator class
#######################################################################

class AxfcTFIRTranslator(AxfcIRTranslator):

    ## The constructor
    def __init__(self, md, path):
        super().__init__(md)

        graph_def = loadFrozenModel(path)
        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def, name='')

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

    ## This method is used get the tensor by name
    # @param self this object
    # @param name tensor's name
    # @return tensor TensorProto object
    def __get_tensor_by_name(self, name: str):

        # check the BiasaddClone
        postfix_name = name.split('/')[-1]
        if postfix_name == 'BiasaddClone':
            name = name.strip('/BiasaddClone')

        return self.graph.get_tensor_by_name('import/' + name + ':0')

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

    ## This method is used to emit the AIXTensor to be the input, output, scale, filter, biase, and variance
    #
    # @param self this object
    # @param attr_tensor AIXTensor object
    # @param isInoutTensor if it is Input or output tensor it must be true
    # @return aix_tensor AIXTensor object
    def __emit_aix_tensor(self, tensor, is_inout_tensor= False) -> AIXLayer.AIXTensor:

        aix_tensor = AIXLayer.AIXTensor()

        # set dtype
        aix_tensor.dtype = aix_data_type_tbl[tensor.dtype]

        if "data_format" in tensor.op.node_def.attr:
            data_format = tensor.op.node_def.attr["data_format"].s
            # set fixed data_format for darknet
            # TODO: delete this line when darknet's data_format support all
            data_format = DEFAULT_TYPE.encode()
        elif tensor.shape.ndims == 1:
            data_format = b'VECTOR'
        else:
            data_format = DEFAULT_TYPE.encode()

        # set format
        aix_tensor.format = aix_tensor_format_tbl[data_format]

        # set fval
        # check if the dataformat is int32 so it uses bval
        dims = print_tensor_content(tensor.op)

        if dims is not None:
            for dim in dims.flatten():
                aix_tensor.fval.append(dim)

        # set dims
        shape = list(map(lambda x: 1 if not x else x, tensor.shape))

        if is_inout_tensor:
            # set None element to 1

            # map 'NHWC' format with opt_shape
            shape_dict = dict(zip('NHWC', shape))

            # reverse appending (following aix compiler structure)
            for t in reversed(DEFAULT_TYPE):
                aix_tensor.dims.append(shape_dict[t])
        else:
            aix_tensor.dims.extend(shape)

        # set size
        aix_tensor.size = np.prod(aix_tensor.dims)

        return aix_tensor

    ## This method is used to set the default hyperparameters: scale, mean, variance
    #
    # @param self this object
    # @param layer AIXLayer object
    # @return AIXTensor it can be mean, scale, or variance tensor
    def __emit_default_hyper_parameter(self, aix_layer: AIXLayer, default_value:int) -> AIXLayer.AIXTensor:

        tensor = AIXLayer.AIXTensor()

        tensor.dtype = aix_layer.output.dtype
        tensor.format = AIXLayer.AIX_FORMAT_VECTOR

        # set the output channel to tensor dims [2]
        output_channel = aix_layer.output.dims[2]
        tensor.dims.append(output_channel)
        tensor.size = output_channel
        tensor.fval.extend([default_value] * output_channel)

        return tensor

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
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        """
        tf.nn.conv2d(
            input, filters, strides, padding, data_format, dilations, name
        )
        """

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
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX group convolution layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX convolution layer
    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_convolution - node %d", ir_node.layer_id)

        """
        tf.nn.conv2d(
            input, filters, strides, padding, data_format, dilations, name
        )
        """

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
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX batchnorm layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX batchnorm layer
    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_batchnorm - node %d", ir_node.layer_id)

        """
        tf.nn.batch_normalization(
            x, mean, variance, offset, scale, variance_epsilon, name=None
        )
        """

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
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX avgpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_avgpool - node %d", ir_node.layer_id)

        """
        tf.nn.avg_pool(
            input, ksize, strides, padding, data_format, name
        )
        """
        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)
        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX maxpool layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_maxpool - node %d", ir_node.layer_id)

        """
        tf.nn.max_pool(
            input, ksize, strides, padding, data_format=None, name=None
        )
        """

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

        # samplingdesc
        sampling_desc = self._emit_aix_sampling_desc(ir_node, tensor=tensor)
        aix_layer.samplingdesc.CopyFrom(sampling_desc)

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX element-wise add (ewadd) layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_ewadd - node %d", ir_node.layer_id)

        """
        tf.math.add(
            x, y, name=None
        )
        """
        # get the aix layer of the given IR node
        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06
        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX softmax layer object. The information includes
    # layer inputs, layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX softmax layer
    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_softmax - node %d", ir_node.layer_id)

        """
        tf.nn.softmax(
            logits, axis=None, name=None
        )
        """
        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX biasadd layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX avgpool layer
    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        logging.info("AxfcTFIRTranslator:_emit_aix_layer_biasadd - node %d", ir_node.layer_id)

        """
        tf.nn.bias_add(
            input, ksize, strides, padding, data_format, name
        )
        """

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter, bias
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))
        aix_layer.bias.CopyFrom(self._emit_aix_tensor_bias(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits some tensorflow-specific information of the given IR node
    # into the given AIX activation layer object. The information includes layer inputs,
    # layer outputs, and so on.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted
    # @return an output AIX activation layer
    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
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

        aix_layer = ir_node.aix_layer

        tensor = self.__get_tensor_by_name(ir_node.name)

        # filter
        aix_layer.filter.CopyFrom(self._emit_aix_tensor_filter(ir_node, tensor=tensor))

        # convolution desc
        aix_layer.convdesc.CopyFrom(self._emit_aix_convolution_desc(ir_node, tensor=tensor))

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

        return AxfcError.SUCCESS

    ##  This method emits an AIX tensor of an input type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an input type
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        # logging.info("AxfcTFIRTranslator:_emit_aix_tensor_input - node %s", ir_node.name)

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

        # get TensorProto object
        tensor = self.__get_tensor_by_name(ir_node.name)

        # get input Tensor
        input_tensors = list(filter(lambda x: x.op.type != 'Const', tensor.op.inputs))
        if input_tensors:
            aix_tensor = self.__emit_aix_tensor(input_tensors[0], is_inout_tensor=True)

        return aix_tensor

    ##  This method emits an AIX tensor of an filter type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an filter type
    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) \
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
        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_filter - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'weights' in tensor.name:
                aix_tensor = self.__emit_aix_tensor(tensor)
                aix_tensor.dims[-1] = ir_node.aix_layer.output.dims[2]
                break

        # set default
        if aix_tensor is None:
            aix_layer = ir_node.aix_layer
            aix_tensor = AIXLayer.AIXTensor()
            aix_tensor.dtype = aix_layer.input.dtype
            aix_tensor.format = aix_layer.input.format
            aix_tensor.dims.append(1)  # width
            aix_tensor.dims.append(1)  # height
            aix_tensor.dims.append(aix_layer.input.dims[2])  # channel
            aix_tensor.dims.append(aix_layer.output.dims[2])  # number of filter

        return aix_tensor

    ##  This method emits an AIX tensor of an bias type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an bias type
    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) \
            -> AIXLayer.AIXTensor:

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_bias - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'biases' in tensor.name or 'beta' in tensor.name:
                aix_tensor = self.__emit_aix_tensor(tensor, )
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits an AIX tensor of an scale type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an scale type
    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, **kwargs) \
            -> AIXLayer.AIXTensor:

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_scale - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'gamma' in tensor.name:
                aix_tensor = self.__emit_aix_tensor(tensor, )
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits an AIX tensor of an mean type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an mean type
    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) \
            -> AIXLayer.AIXTensor:

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_mean - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'moving_mean' in tensor.name:
                aix_tensor = self.__emit_aix_tensor(tensor, )
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits an AIX tensor of an variance type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param is_default indicates if default values are used to emit
    # @return an AIX tensor of an variance type
    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) \
            -> AIXLayer.AIXTensor:

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_tensor_mean - need TensorProto object')

        tensors = [tensor for tensor in tensor.op.inputs]

        aix_tensor = None

        for tensor in tensors:
            if 'moving_variance' in tensor.name:
                aix_tensor = self.__emit_aix_tensor(tensor, )
                break

        # set default
        if aix_tensor is None:
            aix_tensor = self.__emit_default_hyper_parameter(ir_node.aix_layer, 1)

        return aix_tensor

    ##  This method emits an AIX tensor of an output type from the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX tensor of an output type
    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        """
         output[b, i, j, k] =
            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                            filter[di, dj, q, k]
        """

        tensor = self.__get_tensor_by_name(ir_node.name)

        aix_tensor = (self.__emit_aix_tensor(tensor, is_inout_tensor=True))

        return aix_tensor

    ##  This method emits the AIX convolution description of the given IR node.
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @param kwargs keyword arguments used for pass the 'tensor'
    # @return an AIX convolution description
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXConvolutionDesc:

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator: _emit_aix_convolution_desc - need TensorProto object')

        convolution_desc = AIXLayer.AIXConvolutionDesc()

        aix_layer = ir_node.aix_layer

        # dtype
        convolution_desc.dtype = aix_layer.input.dtype

        # strides
        if 'strides' in tensor.op.node_def.attr:
            stride_dict = dict(zip('AHWB', tensor.op.get_attr('strides')))
            convolution_desc.stride.extend([stride_dict['H'], stride_dict['W'],0,0])
        else:
            convolution_desc.stride.extend([1, 1, 0, 0])

        # paddings
        if 'padding' in tensor.op.node_def.attr:

            if tensor.op.get_attr('padding') == b'VALID':
                convolution_desc.padding.extend([0, 0, 0, 0])
            else:  # SAME
                input_h = aix_layer.input.dims[1]
                stride_h = convolution_desc.stride[0]
                filter_h = aix_layer.filter.dims[0]

                input_w = aix_layer.input.dims[0]
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

        # dilation
        if 'dilations' in tensor.op.node_def.attr:
            convolution_desc.dilation.extend(tensor.op.get_attr('dilations'))
        else:
            convolution_desc.dilation.extend([1, 1, 1, 1])

        # groups
        if AIXLayer.AIX_LAYER_GROUP_CONV in aix_layer.type:
            convolution_desc.groups = aix_layer.input.dims[2]
        else:
            convolution_desc.groups = 1

        return convolution_desc

    ##  This method emits the AIX sampling description of the given IR node.
    #  This method must called after _emit_aix_convolution_desc()
    #
    # @param self this object
    # @param ir_node an IR node to be emitted as an AIX tensor
    # @return an AIX sampling description
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXSamplingDesc:

        """
        This method is used in AIXLayer.AIX_LAYER_MAXPOOL,
                                 AIXLayer.AIX_LAYER_AVGPOOL,
                                 AIXLayer.AIX_LAYER_UPSAMPLE,
                                 AIXLayer.AIX_LAYER_REORG
        """

        if 'tensor' in kwargs:
            tensor = kwargs['tensor']
        else:
            logging.error('AxfcTFIRTranslator:_emit_aix_sampling_desc - need TensorProto object')

        sampling_desc = AIXLayer.AIXSamplingDesc()

        # mode
        layer_type = ir_node.aix_layer.type[-1]

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
        else:
            return None

        # strides
        if 'ksize' in tensor.op.node_def.attr:
            stride_dict = dict(zip('AHWB', tensor.op.get_attr('ksize')))
            sampling_desc.window.extend([stride_dict['H'], stride_dict['W'], 0, 0])
        else:
            sampling_desc.window.extend([1, 1, 0, 0])

        # strides
        # TODO: CHECK the strides formula
        # sampling_desc.stride.extend(ir_node.aix_layer.convdesc.stride)
        sampling_desc.stride.extend([0,0,0,0])
        # padding
        sampling_desc.padding.extend(ir_node.aix_layer.convdesc.padding)

        return sampling_desc

