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

import enum
import logging

from AxfcError import *
from AxfcIRBlock import *
from AxfcIRGraph import *
from AxfcMachineDesc import *

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

#######################################################################
# AxfcIRTranslator class
#######################################################################

class AxfcIRTranslator:

    ## @var _md
    # AIX machine description

    ## @var _aix_graphs
    # a list of AIXGraphs translated from an input model

    ## @var _ir_symtab
    # a symbol table of pairs of an IR node's name and itself

    ## @var __emit_aix_layer_tbl
    # a dictionary of pairs of AIXLayerType and its AIX layer emission method

    ## The constructor
    def __init__(self, md):
        self._md = md
        self.aix_graphs = None
        self._ir_symtab = None

        self.__emit_aix_layer_tbl = {
            AIXLayer.AIXLayerType.AIX_LAYER_CONVOLUTION: self._emit_aix_layer_convolution,
            AIXLayer.AIXLayerType.AIX_LAYER_GROUP_CONV: self._emit_aix_layer_group_conv,
            AIXLayer.AIXLayerType.AIX_LAYER_BATCHNORM: self._emit_aix_layer_batchnorm,
            AIXLayer.AIXLayerType.AIX_LAYER_AVGPOOL: self._emit_aix_layer_avgpool,
            AIXLayer.AIXLayerType.AIX_LAYER_BIASADD: self._emit_aix_layer_biasadd,
            AIXLayer.AIXLayerType.AIX_LAYER_ACTIVATION: self._emit_aix_layer_activation
        }

    ## This method translates IR blocks of the given IR graph into AIXGraphs and
    #  return them.
    #
    # @param self this object
    # @param ir_graph input IR graph
    # @return error info and a list of AIXGraphs
    def emit_aixh_graphs(self, ir_graph: AxfcIRGraph) -> {AxfcError, list}:
        logging.info("AxfcIRTranslator:emit_aixh_graph")

        # get the symbol table
        self._ir_symtab = ir_graph.symtab

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
        aix_graph = AIXGraph()

        # translate all the nodes into AIX layers
        for ir_node in ir_block.nodes:
            err, aix_layer = self.__emit_aixh_node(ir_node)
            if err is not AxfcError.SUCCESS:
                return err, None

            # update AIXGraph.layer
            aix_graph.layer.append(aix_layer)

            # update AIXGraph.input_layers
            if ir_node.is_input:
                aix_graph.input_layers.append(aix_layer.id)

            # update AIXGraph.output_layers
            if ir_node.is_output:
                aix_graph.output_layers.append(aix_layer.id)

        return AxfcError.SUCCESS, aix_graph

    ## This method is used to translate an IR node into an AIXLayer object.
    #
    # @param self this object
    # @param ir_node input IR node to be translated
    # @return error info and an output AIXLayer object
    def __emit_aixh_node(self, ir_node: AxfcIRNode) -> {AxfcError, AIXLayer}:
        #logging.info("AxfcTFIRTranslator:_emit_aixh_node - node %d", ir_node.layer_id)

        # get the operation information specified in the machine description
        layer_info = self._md.get_layer_info(ir_node.op)

        # create a new AIX layer
        aix_layer = AIXLayer()

        # layer ID
        aix_layer.id = ir_node.layer_id
        aix_layer.name = ir_node.name

        # layer types
        layer_type = AIXLayer.AIXLayerType.Value(layer_info.layer)
        aix_layer.type.append(layer_type)

        # emit the output specific to each AIX layer type
        try:
            emit_aix_layer = self.__emit_aix_layer_tbl[layer_type]
        except KeyError as e:
            logging.warning(e)
            return AxfcError.UNSUPPORTED_AIX_LAYER_EMIT, None

        ir_node.aix_layer = aix_layer
        err = emit_aix_layer(ir_node)

        if err is not AxfcError.SUCCESS:
            logging.warning("AxfcTFIRTranslator:_emit_aixh_node - "
                            "unsupported layer type %d", layer_type)
            return err, None

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

        # register the generated AIX layer
        ir_node.aix_layer = aix_layer

        return AxfcError.SUCCESS, aix_layer

    ## This method is used to dump AIXGraphs.
    #
    # @param self this object
    def dump_aix_graphs(self):
        logging.info("AxfcIRTranslator:dump_aix_graphs")
        if self.aix_graphs is None:
            logging.warning("No AIXGraphs found")
            return

        str_buf = ""
        for i, aix_graph in enumerate(self.aix_graphs):
            str_buf += ">> AIXGraph: " + str(i) + "\n"
            str_buf += str(aix_graph) + "\n"
        logging.info(str(str_buf))

    def emit_aixh_launcher(self):
        logging.info("AxfcIRTranslator:emit_aixh_launcher")
        pass

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

    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode) -> AxfcError:
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

    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aixh_node(self, ir_node: AxfcIRNode, index: int) -> {AxfcError, AIXLayer}:
        return NotImplementedError()

    ## emission methods for AIX convolution dec
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX sampling dec
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode) -> AIXLayer.AIXTensor:
        return NotImplementedError()
