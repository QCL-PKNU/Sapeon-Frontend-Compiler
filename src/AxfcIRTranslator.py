#######################################################################
#   AxfcIRTranslator
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from aixh_pb2 import *
from AxfcIRGraph import *
from AxfcMachineDesc import *
import tensorflow as tf
from util import *
import json
import numpy as np
from math import ceil

aix_data_type_tbl = {
    tf.float16: AIXLayer.AIXDataType.AIX_DATA_HALF,
    tf.float32: AIXLayer.AIXDataType.AIX_DATA_FLOAT,
    tf.float64: AIXLayer.AIXDataType.AIX_DATA_DOUBLE,
    tf.uint8: AIXLayer.AIXDataType.AIX_DATA_UINT8,
    tf.int8: AIXLayer.AIXDataType.AIX_DATA_SINT8,
    tf.int16: AIXLayer.AIXDataType.AIX_DATA_SINT16
}

aix_tensor_format_tbl = {
    b"NCHW": AIXLayer.AIXTensorFormat.AIX_FORMAT_NCHW,
    b"NHWC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NHWC,
    b"NWHC": AIXLayer.AIXTensorFormat.AIX_FORMAT_NWHC,
    b"VECTOR": AIXLayer.AIXTensorFormat.AIX_FORMAT_VECTOR
}

DEFAULT_TYPE = 'NCHW'


#######################################################################
# AIXInputType enum class
#######################################################################

class AIXTensorType(enum.Enum):
    AIX_TENSOR_INPUT = 0
    AIX_TENSOR_FILTER = 1
    AIX_TENSOR_BIAS = 2
    AIX_TENSOR_SCALE = 3
    AIX_TENSOR_MEAN = 4
    AIX_TENSOR_VARIANCE = 5
    AIX_TENSOR_OUTPUT = 6
    AIX_TENSOR_UNKNOWN = 7

class AIXHyperParam(enum.Enum):
    FILTER = 0
    SCALE = 1
    MEAN = 2
    VARIANCE = 3
    BIAS = 4

#######################################################################
# AxfcIRTranslator class
#######################################################################

class AxfcIRTranslator:

    ## @var _md
    # AIX machine description

    ## @var _calib_data
    # calibration data

    ## @var aix_graphs
    # a list of AIXGraphs translated from an input model

    ## @var _aix_graph
    # the current AIX graph being translated

    ## @var _ir_symtab
    # a symbol table of pairs of an IR node's name and itself

    ## @var __emit_aix_layer_tbl
    # a dictionary of pairs of AIXLayerType and its AIX layer emission method

    ## The constructor
    def __init__(self, md):
        self._md = md
        self.aix_graphs = None
        self._ir_symtab = None
        self._calib_data = None

        self.__emit_aix_layer_tbl = {
            AIXLayer.AIXLayerType.AIX_LAYER_CONVOLUTION: self._emit_aix_layer_convolution,
            AIXLayer.AIXLayerType.AIX_LAYER_GROUP_CONV: self._emit_aix_layer_group_conv,
            AIXLayer.AIXLayerType.AIX_LAYER_BATCHNORM: self._emit_aix_layer_batchnorm,
            AIXLayer.AIXLayerType.AIX_LAYER_AVGPOOL: self._emit_aix_layer_avgpool,
            AIXLayer.AIXLayerType.AIX_LAYER_MAXPOOL: self._emit_aix_layer_maxpool,
            AIXLayer.AIXLayerType.AIX_LAYER_EWADD: self._emit_aix_layer_ewadd,
            AIXLayer.AIXLayerType.AIX_LAYER_SOFTMAX: self._emit_aix_layer_softmax,
            AIXLayer.AIXLayerType.AIX_LAYER_BIASADD: self._emit_aix_layer_biasadd,
            AIXLayer.AIXLayerType.AIX_LAYER_ACTIVATION: self._emit_aix_layer_activation
        }

        model_path = '/Users/hengsengthai/Documents/06. Projects/skt-aix-frontend-compiler/tst/mobilenet_remove_identity.pb'
        graph_def = loadFrozenModel(model_path)
        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def, name='')

    ## This method translates IR blocks of the given IR graph into AIXGraphs and
    #  return them.
    #
    # @param self this object
    # @param ir_graph input IR graph
    # @param calib_data external calibration data
    # @return error info and a list of AIXGraphs
    def emit_aixh_graphs(self, ir_graph: AxfcIRGraph, calib_data: dict) -> {AxfcError, list}:
        logging.info("AxfcIRTranslator:emit_aixh_graph")

        # get the symbol table
        self._ir_symtab = ir_graph.symtab

        # set the calibration data
        self._calib_data = calib_data

        # create a new list of AIX graphs to output
        self.aix_graphs = list()

        # translate all the blocks into AIX graphs
        for ir_block in ir_graph.blocks:
            # ignore blocks not supported by hardware
            if not ir_block.is_aixh_support:
                continue

            err, aix_graph = self.__emit_aixh_block(ir_block)
            if err is AxfcError.SUCCESS:
                self.aix_graphs.append(aix_graph)
                ir_block.aix_graph = aix_graph
            else:
                return err, None

        return AxfcError.SUCCESS, self.aix_graphs

    ## This method is used to translate an IR block into an AIXGraph.
    #
    # @param self this object
    # @param ir_block input IR block
    # @return error info and an output AIXGraph
    def __emit_aixh_block(self, ir_block: AxfcIRBlock) -> {AxfcError, AIXGraph}:
        logging.info("AxfcIRTranslator:__emit_aixh_block - block %d", ir_block.id)

        # create a new AIX graph to output
        self._aix_graph = AIXGraph()

        # translate all the nodes into AIX layers
        for ir_node in ir_block.nodes:

            # skip already emitted nodes
            if ir_node.aix_layer is not None:
                continue

            # emit the current node into an AIX layer and append it to the AIXGraph
            err, aix_layer = self.__emit_aixh_node(ir_node)
            if err is not AxfcError.SUCCESS:
                return err, None

            self._aix_graph.layer.append(aix_layer)

            # update AIXGraph.input_layers
            if ir_node.is_input:
                self._aix_graph.input_layers.append(aix_layer.id)

            # update AIXGraph.output_layers
            if ir_node.is_output:
                self._aix_graph.output_layers.append(aix_layer.id)

        return AxfcError.SUCCESS, self._aix_graph

    ## This method is used to translate an IR node into an AIXLayer object.
    #
    # @param self this object
    # @param ir_node input IR node to be translated
    # @return error info and an output AIXLayer object
    def __emit_aixh_node(self, ir_node: AxfcIRNode) -> {AxfcError, AIXLayer}:
        # logging.info("AxfcTFIRTranslator:_emit_aixh_node - node %d", ir_node.layer_id)

        # get the operation information specified in the machine description
        layer_info = self._md.get_layer_info(ir_node.op)

        # create a new AIX layer
        aix_layer = AIXLayer()

        # layer ID
        aix_layer.id = ir_node.layer_id
        aix_layer.name = ir_node.name

        # layer types
        if not layer_info.is_conv:
            aix_layer.type.append(AIXLayer.AIX_LAYER_SKIP_CONV)
        layer_type = AIXLayer.AIXLayerType.Value(layer_info.layer)
        aix_layer.type.append(layer_type)

        # setup layer: input, output, filter, bias, scale, variance, convdesc, samplingdesc
        self._setup_aix_layer(aix_layer)

        # layer input and output
        if ir_node.is_input:
            aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_INPUT)

        if ir_node.is_output:
            aix_layer.type.append(AIXLayer.AIXLayerType.AIX_LAYER_OUTPUT)

        # predecessors & successors
        for pred in ir_node.preds:
            if pred.is_aixh_support:
                aix_layer.preds.append(pred.layer_id)

        for succ in ir_node.succs:
            if succ.is_aixh_support:
                aix_layer.succs.append(succ.layer_id)

        # activation
        activation = layer_info.activation
        if activation is not None:
            aix_layer.activation = AIXLayer.AIXActivationMode.Value(activation)

        # calibration
        if self._calib_data is not None:
            # get calibration data of this layer
            calib_data = self._calib_data[ir_node.name]
            aix_layer.output_threshold = calib_data["output"]
        else:
            aix_layer.output_threshold = 0

        return AxfcError.SUCCESS, aix_layer

    ## This method is used to setup the aix layer such as:
    # Input, Output, default inputs attributes, filter, scale mean,
    # variance, biase, Convdesc, epsilon
    #
    # @param self this object
    # @param aix_layer is the object of AIXLayer from proto
    def _setup_aix_layer(self, aix_layer):

        tensor = self.graph.get_tensor_by_name('import/' + aix_layer.name + ':0')

        aix_input_tensor_format_type = {
            'gamma': aix_layer.scale,
            'moving_variance': aix_layer.variance,
            'moving_mean': aix_layer.mean,
            'weights': aix_layer.filter,
            # 'beta' : aix_layer.bias,
            'biases': aix_layer.bias
        }

        # Input
        input_tensors = list(filter(lambda x: x.op.type != 'Const', tensor.op.inputs))
        if input_tensors:
            aix_layer.input.CopyFrom(self._emit_input_tensor(input_tensors[0], isInoutTensor=True))

        # Output
        aix_layer.output.CopyFrom(self._emit_input_tensor(tensor, isInoutTensor=True))

        # set default filter
        self._setup_default_hyperparam(aix_layer, AIXHyperParam.FILTER)


        # set default mean, scale , variance to Batchnorm and Conv2
        type = aix_layer.type[-1]
        if type in [AIXLayer.AIX_LAYER_CONVOLUTION, AIXLayer.AIX_LAYER_BATCHNORM]:
            self._setup_default_hyperparam(aix_layer, AIXHyperParam.SCALE)
            self._setup_default_hyperparam(aix_layer, AIXHyperParam.MEAN)
            self._setup_default_hyperparam(aix_layer, AIXHyperParam.VARIANCE)

        # # bias is not in BN or ACT
        # if type not in [AIXLayer.AIX_LAYER_BATCHNORM, AIXLayer.AIX_LAYER_ACTIVATION] :
        #     self.__setup_default_hyperparam(aix_layer, AIXHyperParam.BIAS)

        # filter, scale, mean, variance, bias
        for i in tensor.op.inputs:
            for name, func in aix_input_tensor_format_type.items():
                if name in i.name:
                    func.CopyFrom(self._emit_input_tensor(i))

                    # set the number of filter = output channel
                    if name is 'weights':
                        func.dims[-1] = aix_layer.output.dims[2]

        # Convdesc
        self._setup_convdesc(aix_layer, tensor)

        # samplidesc
        self._setup_samplingdesc(aix_layer)

        # epsilon
        if 'epsilon' in tensor.op.node_def.attr:
            aix_layer.epsilon = tensor.op.node_def.attr['epsilon'].f
        else:
            aix_layer.epsilon = 1e-06

    ## This method is used to emit the AIXTensor to be the input, output, scale, filter, biase, and variance
    #
    # @param self this object
    # @param attr_tensor AIXTensor object
    # @param isInoutTensor if it is Input or output tensor it must be true
    # @return aix_tensor AIXTensor object
    def _emit_input_tensor(self, attr_tensor, isInoutTensor = False) -> AIXLayer.AIXTensor:

        aix_tensor = AIXLayer.AIXTensor()

        # set dtype
        aix_tensor.dtype = aix_data_type_tbl[attr_tensor.dtype]

        if "data_format" in attr_tensor.op.node_def.attr:
            data_format = attr_tensor.op.node_def.attr["data_format"].s
            # set fixed data_format for darknet
            # TODO: delete this line when darknet's data_format support all
            data_format = DEFAULT_TYPE.encode()
        elif attr_tensor.shape.ndims == 1:
            data_format = b'VECTOR'
        else:
            data_format = DEFAULT_TYPE.encode()

        # set format
        aix_tensor.format = aix_tensor_format_tbl[data_format]

        # set fval
        # check if the dataformat is int32 so it uses bval
        dims = print_tensor_content(attr_tensor.op)

        if dims is not None:
            for dim in dims.flatten():
                aix_tensor.fval.append(dim)

        # set dims
        shape = list(map(lambda x: 1 if not x else x, attr_tensor.shape))
        print(attr_tensor.name)
        if isInoutTensor:
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

    ## This method is used to set the default hyperparam: filter, scale, mean, variance
    #
    # @param self this object
    # @param layer AIXLayer object
    # @param hyper_param AIXHyperParam : FILTER, MEAN, SCALE, VARIANCE
    def _setup_default_hyperparam(self, aix_layer: AIXLayer, hyper_param: AIXHyperParam):

        tensor = AIXLayer.AIXTensor()

        # set filter
        if hyper_param is AIXHyperParam.FILTER:

            tensor.dtype = aix_layer.input.dtype
            tensor.format = aix_layer.input.format
            tensor.dims.append(1) # width
            tensor.dims.append(1) # height
            tensor.dims.append(aix_layer.input.dims[2]) # channel
            tensor.dims.append(aix_layer.output.dims[2]) # number of filter

            aix_layer.filter.CopyFrom(tensor)

        else:

            tensor.dtype = aix_layer.output.dtype
            tensor.format = AIXLayer.AIX_FORMAT_VECTOR

            # set the output channel to tensor dim
            output_channel = aix_layer.output.dims[2]
            tensor.dims.append(output_channel)
            tensor.size = output_channel

            # set mean value to 0
            if hyper_param is AIXHyperParam.MEAN:
                tensor.fval.extend([0]*output_channel)
                aix_layer.mean.CopyFrom(tensor)

            # set scale value to 1
            elif hyper_param is AIXHyperParam.SCALE:
                tensor.fval.extend([1] * output_channel)
                aix_layer.scale.CopyFrom(tensor)

            # set variance to 1
            elif hyper_param is AIXHyperParam.VARIANCE:
                tensor.fval.extend([1] * output_channel)
                aix_layer.variance.CopyFrom(tensor)

            # set bias to 1
            elif hyper_param is AIXHyperParam.BIAS:
                tensor.fval.extend([1] * output_channel)
                aix_layer.bias.CopyFrom(tensor)

    ## This method is used to set the convdesc to layer
    #
    # @param layer AIXLayer object
    # @param tensor AIXTensor object
    def _setup_convdesc(self, aix_layer, tensor):

        convdesc = AIXLayer.AIXConvolutionDesc()

        # dtype
        convdesc.dtype = aix_layer.input.dtype

        # strides
        # format from NHWC -> NCHW
        if 'strides' in tensor.op.node_def.attr:
            stride_dict = dict(zip('ABCD', tensor.op.get_attr('strides')))
            for val in 'ADBC':
                convdesc.stride.append(stride_dict[val])
        else:
            convdesc.stride.extend([1, 1, 0, 0])

        # paddings
        if 'padding' in tensor.op.node_def.attr:

            if tensor.op.get_attr('padding') == b'VALID':
                convdesc.padding.extend([0, 0, 0, 0])
            else:  # SAME
                input_h = aix_layer.input.dims[1]
                stride_h = convdesc.stride[0]
                filter_h = aix_layer.filter.dims[0]

                input_w = aix_layer.input.dims[0]
                stride_w = convdesc.stride[1]
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

                convdesc.padding.extend([pad_top, pad_bottom, pad_left, pad_right])

        else:
            convdesc.padding.extend([0, 0, 0, 0])

        # dilation
        if 'dilations' in tensor.op.node_def.attr:
            convdesc.dilation.extend(tensor.op.get_attr('dilations'))
        else:
            convdesc.dilation.extend([1,1,1,1])

        # groups
        if AIXLayer.AIX_LAYER_GROUP_CONV in aix_layer.type:
            convdesc.groups = aix_layer.input.dims[2]
        else:
            convdesc.groups = 1

        aix_layer.convdesc.CopyFrom(convdesc)

    ## This method is used to set the samplingdesc to layer
    def _setup_samplingdesc(self, aix_layer):

        if len(aix_layer.type) > 1:

            if aix_layer.type[1] in [AIXLayer.AIX_LAYER_MAXPOOL,
                                 AIXLayer.AIX_LAYER_AVGPOOL,
                                 AIXLayer.AIX_LAYER_UPSAMPLE,
                                 AIXLayer.AIX_LAYER_REORG]:

                samplingdesc = AIXLayer.AIXSamplingDesc()
                samplingdesc.mode = AIXLayer.AIX_POOLING_AVERAGE
                samplingdesc.padding.extend([0, 0, 0, 0])
                samplingdesc.stride.extend([0, 0, 0, 0])
                samplingdesc.window.extend([0, 0, 0, 0])

                aix_layer.samplingdesc.CopyFrom(samplingdesc)

    ## This method is used to return a list of already emitted input nodes.
    #  If there are input nodes that have not translated yet,
    #  we perform __emit_aixh_node method repeatedly to emit them all.
    #
    # @param self this object
    # @param ir_node current node to emit its input nodes
    # @return a list of emitted input nodes
    def _get_emitted_input_nodes(self, ir_node: AxfcIRNode) -> {AxfcError, list}:
        # logging.info("AxfcTFIRTranslator:_get_emitted_input_nodes - node %d", ir_node.layer_id)

        # input nodes
        input_nodes = list()

        for input_node in ir_node.preds:

            if not input_node.is_aixh_support:
                input_nodes.append(input_node)
                continue

            # emit the nodes that have not emitted yet
            if input_node.aix_layer is None:
                err, aix_layer = self.__emit_aixh_node(input_node)

                if err != AxfcError.SUCCESS:
                    logging.warning("AxfcIRTranslator: _get_input_aix_layers - %s", err)
                    return err, None

                self._aix_graph.layer.append(aix_layer)

            # append the emitted node into a list
            input_nodes.append(input_node)

        return AxfcError.SUCCESS, input_nodes

    ## For debugging
    def __str__(self):
        pass

    #######################################################################
    ## Abstract methods
    #######################################################################

    ## emission methods for AIX layers
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode) -> AxfcError:
        return NotImplementedError()

    ## emission methods for AIX tensors
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, output_dims: list) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX convolution dec
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX sampling dec
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()
