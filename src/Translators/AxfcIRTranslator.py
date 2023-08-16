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

import sys
sys.path.append("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/src/")

from aixh_pb2 import *
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
                
                #set input/output to the graph
                for node in ir_block.nodes:
                    if node.is_input:
                        aix_graph.input_layers.append(node.layer_id)
                    elif node.is_output:
                        aix_graph.output_layers.append(node.layer_id)

                self.aix_graphs.append(aix_graph)
                ir_block.aix_graph = aix_graph
            else:
                return err, None

        return AxfcError.SUCCESS, self.aix_graphs

    ## This method is used to translate an IR block into an AIXGraph.
    # @param self this object
    # @param ir_block input IR block
    # @return error info and an output AIXGraph
    def __emit_aixh_block(self, ir_block: AxfcIRBlock) -> {AxfcError, AIXGraph}:
        logging.info("AxfcIRTranslator:__emit_aixh_block - block %d", ir_block.id)

        # create a new AIX graph to output
        self._aix_graph = AIXGraph()

        # translate all the nodes into AIX layers
        for ir_node in ir_block.nodes:

            # skip already emitted nodes and Const
            if ir_node.aix_layer is not None or ir_node.op == "Const":
                continue

            # emit the current node into an AIX layer and append it to the AIXGraph
            err, aix_layer = self.__emit_aixh_node(ir_node, ir_block)
            if err is not AxfcError.SUCCESS:
                return err, None

            self._aix_graph.layer.append(aix_layer)

            # SENGTHAI: we can omit input_layers / output_layers
            # because after merge, the merger will generate it automatically.
            # # update AIXGraph.input_layers
            # if ir_node.is_input:
            #     self._aix_graph.input_layers.append(aix_layer.id)
            #
            # # update AIXGraph.output_layers
            # if ir_node.is_output:
            #     self._aix_graph.output_layers.append(aix_layer.id)

        return AxfcError.SUCCESS, self._aix_graph

    ## This method is used to translate an IR node into an AIXLayer object.
    #
    # @param self this object
    # @param ir_node input IR node to be translated
    # @return error info and an output AIXLayer object
    def __emit_aixh_node(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> {AxfcError, AIXLayer}:
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
        
        #check if ir_node is the block input & output
        if ir_node.is_input:
            input_layer = AIXLayer.AIXLayerType.Value("AIX_LAYER_INPUT")
            aix_layer.type.append(input_layer)
        elif ir_node.is_output:
            output_layer = AIXLayer.AIXLayerType.Value("AIX_LAYER_OUTPUT")
            aix_layer.type.append(output_layer)

        # Emit the input tensor of node, not the block input
        aix_layer.input.CopyFrom(self._emit_aix_tensor_input(ir_node))

        # Emit the output tensor of node, not the block output
        # logging.warning("AxfcIRTranslator: AIXLayer output can be multiple layers.")
        aix_layer.output.CopyFrom(self._emit_aix_tensor_output(ir_node))

        # emit the output specific to each AIX layer type
        try:
            emit_aix_layer = self.__emit_aix_layer_tbl[layer_type]
        except KeyError:
            logging.warning("__emit_aixh_node: unsupported layer type - %s, %s", ir_node.op, layer_type)
            return AxfcError.UNSUPPORTED_AIX_LAYER_EMIT, None

        # register the generated AIX layer
        ir_node.aix_layer = aix_layer

        # perform the emission of AIX layers
        err = emit_aix_layer(ir_node)

        if err is not AxfcError.SUCCESS:
            logging.warning("AxfcIRTranslator:__emit_aixh_node - %s", err)
            return err, None

        # predecessors & successors
        for pred in ir_node.preds:
            if pred.is_aixh_support and pred in ir_block.nodes:
                aix_layer.preds.append(pred.layer_id)

        for succ in ir_node.succs:
            if succ.is_aixh_support and succ in ir_block.nodes:
                aix_layer.succs.append(succ.layer_id)

        # activation
        activation = layer_info.activation
        if activation is not None:
            aix_layer.activation = AIXLayer.AIXActivationMode.Value(activation)

        # calibration
        if self._calib_data is not None:
            # get calibration data of this layer
            postfix_name = ir_node.name.split('/')[-1]
            name = ir_node.name
            if postfix_name == 'BiasaddClone':
                name = ir_node.name.replace('/BiasaddClone', '')
            
            calib_data = self._calib_data.get(name)

            if calib_data:
                aix_layer.output_threshold = calib_data["output"]
                aix_layer.input_threshold = calib_data["input"]
            else:
                logging.warning("AxfcIRTranslator: {} - {}".format(name, "No the calibration data"))

            if ir_node.is_input:
                aix_layer.input_threshold = self._calib_data[list(self._calib_data)[0]]["input"]

        else:
            aix_layer.output_threshold = 0

        return AxfcError.SUCCESS, aix_layer

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
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    ## emission methods for AIX tensors
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX convolution dec
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX sampling dec
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()
