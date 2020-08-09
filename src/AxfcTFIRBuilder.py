#######################################################################
#   AxfcTFIRBuilder
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

from AxfcError import *
from AxfcIRGraph import *
from AxfcIRBuilder import *
from AxfcGraphUtil import *

#######################################################################
# AxfcTFIRBuilder class
#######################################################################
class AxfcTFIRBuilder(AxfcIRBuilder):

    ## The constructor
    def __init__(self, md):
        super().__init__(md)

    ## This method is used to read a tensorflow graph from an input file in the given path.
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _read_model_graph(self, path: str) -> AxfcError:
        logging.info("AxfcTFIRBuilder:read_model_graph - path: %s", path)

        # read input Tensorflow graph_def
        self._tf_graph = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(path, 'rb') as f:
            self._tf_graph.ParseFromString(f.read())
            tf.import_graph_def(self._tf_graph)

        return AxfcError.SUCCESS

    ## This method is used to construct a naive AIXIR using a tensorflow graph.
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _build_naive_ir(self, path: str) -> {AxfcError}:
        logging.info("AxfcTFIRBuilder:build_naive_ir - path: %s", path)

        # read a Tensorflow graph
        err = self._read_model_graph(path)
        if err != AxfcError.SUCCESS:
            return err

        # translation from Tensorflow graph to AIXIR
        tf_graph_def = self._tf_graph

        if tf_graph_def is None:
            return AxfcError.INVALID_TF_GRAPH

        # build AIX IR graph using the nodes of the Tensorflow graph
        self._ir_graph = AxfcIRGraph()

        for tf_node_def in tf_graph_def.node:
            self.__append_node_def(tf_node_def)

        return AxfcError.SUCCESS

    ## This method is used to create a new IR node from tf.NodeDef and append it to the IR graph.
    #  The successors and predecessors of the IR node is found using the symbol table.
    # @param self this object
    # @param tf_node_def input node_def object of Tensorflow
    # @return error info.
    def __append_node_def(self, tf_node_def: tf.compat.v1.NodeDef) -> AxfcError:
        #logging.info("AxfcTFIRBuilder:append_node_def - tf_node_def: %s", tf_node_def.name)

        # create a new IR node
        ir_node = AxfcIRNode(tf_node_def)

        # register the IR node to the symbol table with the name of node_def
        self._ir_symtab[tf_node_def.name] = ir_node

        # set the operation of this node
        ir_node.op = tf_node_def.op

        # check the node that is supported by AIXH hardware
        if self._md.get_axih_support(ir_node.op):
            ir_node.is_aixh_support = True
        else:
            ir_node.is_aixh_support = False

        # connect predecessors and successor
        for pred_name in tf_node_def.input:

            # find the predecessor using the symbol table
            pred_node = self._ir_symtab[pred_name]

            if pred_node is not None:
                pred_node.succs.append(ir_node)
                ir_node.preds.append(pred_node)
            else:
                return AxfcError.PRED_NODE_NOT_FOUND

        # append the new IR into a IR graph
        self._ir_graph.append_node(ir_node)

        return AxfcError.SUCCESS

    ## This method is used to visualize the IR graph using Sigma js.
    # @param self this object
    def _visualize_graph(self):

        graph = AxfcGraphUtil()
        
        # Nested function to ignore edges from a constant node
        def is_ignored(op:str) -> bool:
            return (op == 'Const' or op == 'Identity')
        
        # build a AxfcGraph
        for ir_node in self._ir_graph.nodes:
            
            # ignore some edges
            if is_ignored(ir_node.op):
                continue

            graph.add_node(ir_node)

            for succ in ir_node.succs:
                # ignore some edges
                if is_ignored(succ.op):
                    continue

                graph.add_edge(ir_node.id, succ.id)
                
        graph.build()
        graph.show()

    ## For debugging
    def __str__(self):
        #tf.io.write_graph(self._tf_graph, './', '../tst/graph_def.pbtxt', as_text=True)
        return
