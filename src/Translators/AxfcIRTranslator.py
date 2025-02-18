#######################################################################
#   AxfcIRTranslator
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################

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
    """
    A class for translating input models into AIXGraphs based on AIX machine descriptions.

    Attributes:
        _md: AIX machine description.
        _calib_data: Calibration data for the translation process.
        aix_graphs: A list of AIXGraphs translated from an input model.
        _aix_graph: The current AIX graph being translated.
        _ir_symtab: A symbol table of pairs of an IR node's name and itself.
        __emit_aix_layer_tbl: A dictionary of pairs of AIXLayerType and its AIX layer emission method.
    """

    def __init__(self, md):
        self._md = md
        self.aix_graphs = None
        self._ir_symtab = None
        self._calib_data = None

        self.__emit_aix_layer_tbl = {
            AIXLayer.AIXLayerType.AIX_LAYER_CONVOLUTION: self._emit_aix_layer_convolution,
            AIXLayer.AIXLayerType.AIX_LAYER_GROUP_CONV: self._emit_aix_layer_group_conv,
            AIXLayer.AIXLayerType.AIX_LAYER_BATCHNORM: self._emit_aix_layer_batchnorm,
            # AIXLayer.AIXLayerType.AIX_LAYER_DOWNSAMPLE: self._emit_aix_layer_downsample,
            AIXLayer.AIXLayerType.AIX_LAYER_AVGPOOL: self._emit_aix_layer_avgpool,
            AIXLayer.AIXLayerType.AIX_LAYER_MAXPOOL: self._emit_aix_layer_maxpool,
            AIXLayer.AIXLayerType.AIX_LAYER_EWADD: self._emit_aix_layer_ewadd,
            AIXLayer.AIXLayerType.AIX_LAYER_SOFTMAX: self._emit_aix_layer_softmax,
            AIXLayer.AIXLayerType.AIX_LAYER_BIASADD: self._emit_aix_layer_biasadd,
            AIXLayer.AIXLayerType.AIX_LAYER_ACTIVATION: self._emit_aix_layer_activation
        }


    def emit_aixh_graphs(self, ir_graph: AxfcIRGraph, calib_data: dict):
        """Translates IR blocks into AIXGraphs using provided calibration data.

        Args:
            ir_graph (AxfcIRGraph): The input IR graph containing blocks to be translated.
            calib_data (dict): External calibration data for optimizing the translation.

        Returns:
            Tuple[AxfcError, List[AIXGraph]]: An error code and a list of generated AIXGraphs.
        """
        logging.info("AxfcIRTranslator:emit_aixh_graph")

        self._ir_symtab = ir_graph.symtab
        self._calib_data = calib_data
        self.aix_graphs = list()

        for block in ir_graph.blocks:
            if not block.is_aixh_support:
                continue

            err, aix_graph = self.__emit_aixh_block(block)
            if err is AxfcError.SUCCESS:
                
                # todo: set input and output to the graph
                for node in block.nodes:
                    if node.is_input:
                        aix_graph.input_layers.append(node.layer_id)
                    elif node.is_output:   
                        aix_graph.output_layers.append(node.layer_id)

                self.aix_graphs.append(aix_graph)
                block.aix_graph = aix_graph
            else:
                return err, None
            
        return AxfcError.SUCCESS, self.aix_graphs


    def __emit_aixh_block(self, ir_block: AxfcIRBlock) -> {AxfcError, AIXGraph}:
        """Translates an IR block into an AIXGraph.

        Args:
            ir_block (AxfcIRBlock): The IR block to be translated.

        Returns:
            Tuple[AxfcError, AIXGraph]: Error code and the resulting AIXGraph object.
        """
        logging.info(f'AxfcIRTranslator:__emit_aixh_block from block with id: {ir_block.id}')
        self._aix_graph = AIXGraph()

        for node in ir_block.nodes:
            if node.aix_layer is not None or node.op == 'Const':
                continue

            err, aix_layer = self.__emit_aixh_node(node, ir_block)
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


    def __emit_aixh_node(self, ir_node: AxfcIRNode, ir_block: AxfcIRBlock) -> {AxfcError, AIXLayer}:
        """Translates an IR node into an AIXLayer object.

        Args:
            ir_node (AxfcIRNode): The IR node to be translated.
            ir_block (AxfcIRBlock): The IR block containing the node.

        Returns:
            Tuple[AxfcError, AIXLayer]: Error code and the resulting AIXLayer object.
        """

        # logging.info("AxfcTFIRTranslator:_emit_aixh_node - node %d", ir_node.layer_id)

        logging.info(f'AxfcIRTranslator:__emit_aixh_node - node {ir_node.layer_id}')

        layer_info = self._md.get_layer_info(ir_node.op)
        aix_layer = AIXLayer()
        aix_layer.id = ir_node.layer_id
        aix_layer.name = ir_node.name

        # Layer types
        if not layer_info.is_conv:
            aix_layer.type.append(AIXLayer.AIX_LAYER_SKIP_CONV)
        layer_type = AIXLayer.AIXLayerType.Value(layer_info.layer)
        aix_layer.type.append(layer_type)

        # Check if the node is the block's input & output
        if ir_node.is_input:
            input_layer = AIXLayer.AIXLayerType.Value("AIX_LAYER_INPUT")
            aix_layer.type.append(input_layer)
        elif ir_node.is_output:
            output_layer = AIXLayer.AIXLayerType.Value("AIX_LAYER_OUTPUT")
            aix_layer.type.append(output_layer)

        # todo: emit the input and output tensor of the node
        aix_layer.input.CopyFrom(self._emit_aix_tensor_input(ir_node))
        aix_layer.output.CopyFrom(self._emit_aix_tensor_output(ir_node))


        # todo: emit the output based on each layer type
        emit_aix_layer = None
        try:
            emit_aix_layer = self.__emit_aix_layer_tbl[layer_type]
        except:
            logging.warning(f"__emit_aixh_node: unsupported layer type - {ir_node.op}, {layer_type}")
            return AxfcError.UNSUPPORTED_AIX_LAYER_EMIT, None


        # Register aix_layer into the IRNode object
        ir_node.aix_layer = aix_layer

        err = emit_aix_layer(ir_node)
        if err is not AxfcError.SUCCESS:
            logging.warning("AxfcIRTranslator:__emit_aixh_node - %s", err)
            return err, None
        
        for pred in ir_node.preds:
            if pred.is_aixh_support and pred in ir_block.nodes:
                aix_layer.preds.append(pred.layer_id)
        
        for succ in ir_node.succs:
            if succ.is_aixh_support and succ in ir_block.nodes:
                aix_layer.succs.append(succ.layer_id)

        # Activation
        activation = layer_info.activation
        if activation is not None:
            aix_layer.activation = AIXLayer.AIXActivationMode.Value(activation)
        

        # todo: perform calibration
        # if self._calib_data is not None:
        #     # get calibration data of this layer
        #     postfix_name = ir_node.name.split('/')[-1]
        #     name = ir_node.name
        #     if postfix_name == 'BiasaddClone':
        #         name = ir_node.name.replace('/BiasaddClone', '')
            
        #     calib_data = self._calib_data.get(name)

        #     if calib_data:
        #         aix_layer.output_threshold = calib_data["output"]
        #         aix_layer.input_threshold = calib_data["input"]
        #     else:
        #         logging.warning("AxfcIRTranslator: {} - {}".format(name, "No the calibration data"))

        #     if ir_node.is_input:
        #         aix_layer.input_threshold = self._calib_data[list(self._calib_data)[0]]["input"]

        # else:
        #     aix_layer.output_threshold = 0
        aix_layer.output_threshold = 0

        return AxfcError.SUCCESS, aix_layer


    def _get_emitted_input_nodes(self, ir_node: AxfcIRNode) -> {AxfcError, list}:
        """
        Retrieves a list of already emitted input nodes of a given IR node. If there are input nodes 
        that haven't been translated yet, they are emitted using the __emit_aixh_node method.

        Args:
            ir_node (AxfcIRNode): The IR node whose input nodes are to be emitted.

        Returns:
            Tuple[AxfcError, List[AxfcIRNode]]: Error code and a list of emitted input nodes.
        """
        # logging.info("AxfcTFIRTranslator:_get_emitted_input_nodes - node %d", ir_node.layer_id)

        input_nodes = list()

        for input_node in ir_node.preds:
            # Skip nodes not supported by AIX hardware
            if not input_node.is_aixh_support:
                input_nodes.append(input_node)
                continue

            # Emits the nodes that haven't been translated yet
            if input_node.aix_layer is None:
                err, aix_layer = self.__emit_aixh_node(input_node)
                if err != AxfcError.SUCCESS:
                    logging.warning("AxfcIRTranslator: _get_input_aix_layers - %s", err)
                    return err, None

                self._aix_graph.layer.append(aix_layer)

            # Includes the emitted node into a list
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
