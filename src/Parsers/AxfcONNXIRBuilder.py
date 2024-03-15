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

import onnx 
from onnx import helper, shape_inference, save_model
# import onnx_graphsurgeon as gs

import logging
from AxfcError      import AxfcError
from . import AxfcIRBuilder
from AxfcIRNode     import AxfcIRNode

import util.AxfcUtil as _util

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
        self.__initializer_dict = dict()
        self._ir_symtab = dict()
        self.model_input = None

    ## This method is used to read a onnx graph from an input file in the given path.
    #
    # @param self this object
    # @param path file path of input neural network model
    # @return error info
    def _read_model_graph(self, path: str):

        #write log
        logging.info("AxfcONNXIRBuilder:read_model_graph - path: %s", path)

        # read input ONNX graph
        # ex) load("model.onnx")
        onnx_model = onnx.load(path)

        inferred_model = shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(inferred_model)
        print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

        #remove training nodes
        #use onnx runtime for removing

        self.__onnx_graph = onnx_model.graph
        # print(self.__onnx_graph)

        return AxfcError.SUCCESS


    ## This method is used to construct a navie AIXIR using a onnx graph.
    #
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _build_naive_ir(self, path: str) -> AxfcError:
        
        #read onnx graph
        err = self._read_model_graph(path)

        if err is not AxfcError.SUCCESS:
            return err
        
        #translate from ONNX graph to AIXIR
        onnx_graph_def = self.__onnx_graph

        if onnx_graph_def is None:
            #Should respond INVALID_ONNX_GRAPH
            return AxfcError.INVALID_IR_GRAPH

        #Append to _ir_symtab_op op for ops, inputs, outputs
        #Append to _ir_symtab_cnt for constants from initializer

        #build ir node for constants:
        for onnx_node_def in onnx_graph_def.initializer:
            # print("def", onnx_node_def)

            #append ir node into the _ir_symtab
            err = self.__append_node_sym_ir(onnx_node_def, op="Const")
            if err is not AxfcError.SUCCESS:
                return err

        #build ir node for input
        for count, onnx_node_def in enumerate(onnx_graph_def.input):
            #Model input
            if count == 0:
                err = self.__append_node_sym_ir(onnx_node_def, op="Input")
            else:
                #append ir node into the _ir_symtab
                err = self.__append_node_sym_ir(onnx_node_def, op="Const")
            
            if err is not AxfcError.SUCCESS:
                return err
        
        #build ir node
        for onnx_node_def in onnx_graph_def.node:
            #append ir node into the _ir_symtab
            err = self.__append_node_sym_ir(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err
            
            #append ir node def
            err = self.__append_node_def(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err

        ## To make the connection from each nodes, connect a pred/succ
        # Connect the operator node
        for onnx_node_def in onnx_graph_def.node:
            #build node connection
            err = self.__connect_node_def(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err
        
        # Connect the input node
        for onnx_node_def in onnx_graph_def.input:
            #build node connection
            err = self.__connect_node_def(onnx_node_def)
            if err is not AxfcError.SUCCESS:
                return err
        
        # Connect the initializer node
        for onnx_node_def in onnx_graph_def.initializer:
            #build node connection
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
    def __append_node_sym_ir(self, onnx_node_def, op = None) -> AxfcError:
        #initializing ir node 
        ir_node         = AxfcIRNode(onnx_node_def)
        ir_node.name    = onnx_node_def.name

        #set op
        if op:
            ir_node.op = op
        
        #ops
        self._ir_symtab[onnx_node_def.name] = ir_node

        return AxfcError.SUCCESS
    
    ## This method is used to create a new IR node from onnx_node_def and append it to the IR graph.
    # The successors and predecessors of the IR node is found using the symbolic table.
    #
    # @param self this object
    # @param onnx_node_def node definition of onnx model
    # @return erro info
    def __append_node_def(self, onnx_node_def) -> AxfcError:
        
        #get ir node
        ir_node = self._ir_symtab.get(onnx_node_def.name)

        # set ir node operation
        if ir_node.op is None:
            ir_node.op = onnx_node_def.op_type
        
        #check the node that is supported by AIXH hardware
        layer_info = self._md.get_layer_info(ir_node.op)

        if self._md.get_aixh_support(ir_node.op) and not self._md.BREAK_POINT_CONDITION:
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
    def __connect_node_def(self, onnx_node_def) -> AxfcError:

        #get ir node
        ir_node = self._ir_symtab.get(onnx_node_def.name)

        if not ir_node:
            logging.error("AxfcONNXIRBuilder:_connect_node_def ir_node: %s not found", onnx_node_def.name)
        
        #Const and input are not required to connect
        if ir_node.op != None and ir_node.op not in ["Const", "Input"]:

            for pred_name in onnx_node_def.input:
                
                #get ir node
                pred_node = self._ir_symtab.get(pred_name)
                if not pred_node:
                    logging.error("AxfcONNXIRBuilder:_connect_node_def ir_node: %s pred not found", onnx_node_def.name)
                    return AxfcError.PRED_NODE_NOT_FOUND
                
                pred_node.succs.append(ir_node)
                ir_node.preds.append(pred_node)
        
        #add node to ir graph
        self._ir_graph.append_node(ir_node)

        return AxfcError.SUCCESS
        


        
        