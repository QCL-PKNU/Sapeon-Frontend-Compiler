#######################################################################
#   AxfcONNXIRBuilder
#
#   Created: 2022. 01. 05
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Hour Leanghok (leanghok@pukyong.ac.kr)
#
#   Quantum Computing Laboratory (quantum.pknu.ac.kr)
#######################################################################

import logging
import onnx

from onnx import shape_inference
from AxfcError import AxfcError
from .AxfcIRBuilder import AxfcIRBuilder
from AxfcIRNode     import AxfcIRNode

#######################################################################
# AxfcONNXIRBuilder class
#######################################################################

class AxfcONNXIRBuilder(AxfcIRBuilder):

    ## @var _onnx_graph
    # input onnx graph

    ## @var _ir_symtab
    # symbolic table for IR graph

    ## @var model_input
    # input onnx model

    ##The constructor
    def __init__(self, md):
        super().__init__(md)

        self.__onnx_graph = None
        self._ir_symtab = dict()
        self.model_input = None

    ## This method is used to read a onnx graph from an input file in the given path.
    #
    # @param self this object
    # @param path file path of input neural network model
    # @return error info
    def _read_model_graph(self, path: str):
        logging.info("AxfcONNXIRBuilder:read_model_graph - path: %s", path)

        onnx_model = onnx.load(path)
        inferred_model = shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(inferred_model)
        # print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

        #remove training nodes
        #use onnx runtime for removing

        self.__onnx_graph = onnx_model.graph

        if self.__onnx_graph is None:
            return AxfcError.INVALID_ONNX_GRAPH
        # print(self.__onnx_graph)

        return AxfcError.SUCCESS


    ## This method is used to construct a navie AIXIR using a onnx graph.
    #
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _build_naive_ir(self, path: str) -> AxfcError:

        err = self._read_model_graph(path)
        if err is not AxfcError.SUCCESS:
            return err
        
        graph_def = self.__onnx_graph
        
        # Build IRNode for Const
        for node_def in graph_def.initializer:
            err = self.__append_node_sym_ir(node_def, op='Const')
            if err is not AxfcError.SUCCESS:
                return err
            
        for idx, node_def in enumerate(graph_def.input):
            if idx == 0:
                err = self.__append_node_sym_ir(node_def, op="Input")
            else:
                err = self.__append_node_sym_ir(node_def, op="Const")

            if err is not AxfcError.SUCCESS:
                return err
            
        
        # Build IRNode for graph nodes
        for node_def in graph_def.node:
            err = self.__append_node_sym_ir(node_def)
            if err is not AxfcError.SUCCESS:
                return err
            
            err = self.__append_node_def(node_def)
            if err is not AxfcError.SUCCESS:
                return err
            
        # Connect nodes considering preds and succs
        for idx, node_def in enumerate(graph_def.node):
            err = self.__connect_node_def(node_def)
            if err is not AxfcError.SUCCESS:
                return err
        
        # Connect the input node
        for onnx_node_def in graph_def.input:
            #build node connection
            err = self.__connect_node_def(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err
        
        # Connect the initializer node
        for onnx_node_def in graph_def.initializer:
            err = self.__connect_node_def(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err
                    
        return AxfcError.SUCCESS


    ## This method is used to append ir_node into ir symbolic table.
    #
    # @param self this object
    # @param onnx_node_def node definition of onnx model
    # @param op operator
    # @return error info
    def __append_node_sym_ir(self, node_def, op = None) -> AxfcError:
        if node_def is None:
            logging.warning("AxfcONNXIRBuilder: Failed to append node into symbolic IR")
            return AxfcError.INVALID_NODE_DEF

        ir_node = AxfcIRNode(node_def)
        ir_node.name = node_def.name

        if hasattr(node_def, 'output'):
            ir_node.output_name = node_def.output

        if op:
            ir_node.op = op
            
        self._ir_symtab[node_def.name] = ir_node

        return AxfcError.SUCCESS
    

    ## This method is used to create a new IR node from onnx_node_def and append it to the IR graph.
    # The successors and predecessors of the IR node is found using the symbolic table.
    #
    # @param self this object
    # @param onnx_node_def node definition of onnx model
    # @return erro info
    def __append_node_def(self, node_def) -> AxfcError:
        
        ir_node = self._ir_symtab.get(node_def.name)

        if ir_node.op is None:
            ir_node.op = node_def.op_type

        layer_info = self._md.get_layer_info(ir_node.op)
        hardware_support = self._md.get_aixh_support(ir_node.op)

        if hardware_support and not self._md.BREAK_POINT_CONDITION:
            ir_node.is_aixh_support = True
            ir_node.aixh_profit = layer_info.profit
        else:
            ir_node.is_aixh_support = False
            ir_node.aixh_profit = 0

        return AxfcError.SUCCESS
    

    ## This method is used to connect the IR node considering their successors and predecessors.
    #
    # @param self this object
    # @onnx_node_def node definition of onnx model
    # @return error info
    def __connect_node_def(self, node_def) -> AxfcError:

        node = self._ir_symtab.get(node_def.name)
        if not node:
            logging.error("_connect_node_def ir_node: %s not found", node_def.name)
            return AxfcError.PRED_NODE_NOT_FOUND

        # Preserve the original hardware support status, name, and operation type
        hardware_support = node.is_aixh_support
        node_name = node.name
        node_op = node.op

        # Const and input nodes are not required to connect
        if node.op and node.op not in ["Const", "Input"]:
            for idx, pred_name in enumerate(node_def.input):
                pred_node = self._ir_symtab.get(pred_name)
                if pred_node:
                    pred_node.succs.append(node)
                    node.preds.append(pred_node)
                else:
                    found = False
                    for _, potential_node in self._ir_symtab.items():
                        if pred_name in potential_node.output_name:
                            pred_node = potential_node
                            pred_node.succs.append(node)
                            node.preds.append(pred_node)
                            found = True
                            break
                    if not found:
                        logging.error("Pred node: %s not found", pred_name)
                        return AxfcError.PRED_NODE_NOT_FOUND

        # Restore the original hardware support status, name, and operation type
        node.is_aixh_support = hardware_support
        node.name = node_name
        node.op = node_op

        # Add node to IR graph
        self._ir_graph.append_node(node)
        return AxfcError.SUCCESS


        
        