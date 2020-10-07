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

from AxfcIRBuilder import *
from AxfcGraphWriter import *


#######################################################################
# AxfcTFIRBuilder class
#######################################################################

class AxfcTFIRBuilder(AxfcIRBuilder):

    ## The constructor
    def __init__(self, md):
        super().__init__(md)

    ## This method is used to read a tensorflow graph from an input file in the given path.
    #
    # @param self this object
    # @param path file path of input network model
    # @return error info
    def _read_model_graph(self, path: str) -> AxfcError:
        logging.info("AxfcTFIRBuilder:read_model_graph - path: %s", path)

        # read input Tensorflow graph_def
        tf_graph = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(path, 'rb') as f:
            tf_graph.ParseFromString(f.read())
            tf.import_graph_def(tf_graph)

        # remove identity nodes
        # self._tf_graph = tf.compat.v1.graph_util.remove_training_nodes(tf_graph)
        self._tf_graph = tf_graph

        return AxfcError.SUCCESS

    ## This method is used to construct a naive AIXIR using a tensorflow graph.
    #
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
        for tf_node_def in tf_graph_def.node:

            # Process separate: BN -> BN + BiasAdd
            for index, pred_name in enumerate(tf_node_def.input):
                # find the predecessor using the symbol table
                if not (pred_name in self._ir_symtab):
                    continue

                pred_node = self._ir_symtab[pred_name]
                if pred_node is not None:
                    if pred_node.op == 'FusedBatchNorm':
                        tf_node_def.input[index] += '/BiasaddClone'

            err = self.__append_node_def(tf_node_def)

            if tf_node_def.op == 'FusedBatchNorm':
                tf_node_clone_def = tf.compat.v1.NodeDef()
                tf_node_clone_def.CopyFrom(tf_node_def)

                # clear all elements in input
                tf_node_clone_def.input[:] = []

                tf_node_clone_def.input.append(tf_node_def.name)
                tf_node_clone_def.name += '/BiasaddClone'
                tf_node_clone_def.op = 'BiasAdd'

                err = self.__append_node_def(tf_node_clone_def)
            # end process separate

            # end process separate


            if err != AxfcError.SUCCESS:
                return err

        # remove unnecessary IR nodes
        return self.__prune_ir_nodes()

    ## This method is used to prune unnecessary nodes from the IR graph.
    #  Currently, we will remove "identity" and "pad" nodes for the IR translation.
    #
    # @param self this object
    # @return error info
    def __prune_ir_nodes(self) -> AxfcError:

        # for each node
        for ir_node in self._ir_graph.nodes:

            if ir_node.op == "Identity":
                # check validity of the identity node
                if len(ir_node.preds) != 1 or len(ir_node.succs) != 1:
                    return AxfcError.INVALID_IDENTITY_LAYER

                # remove the current node from the graph
                pred_node = ir_node.preds[0]
                succ_node = ir_node.succs[0]

                pred_node.succs[0] = succ_node

                for i, node in enumerate(succ_node.preds):
                    if node == ir_node:
                        succ_node.preds[i] = pred_node
                        break

                self._ir_graph.nodes.remove(ir_node)

            elif ir_node.op == "Pad":
                # check validity of the pad node
                if len(ir_node.preds) != 2 or len(ir_node.succs) != 1:
                    return AxfcError.INVALID_PAD_LAYER

                # check the following convolution node
                succ_node = ir_node.succs[0]

                if succ_node.op.find("Conv") < 0:
                    return AxfcError.INVALID_PAD_LAYER

                # remove the current node from the graph
                pred_node = ir_node.preds[0]

                pred_node.succs[0] = succ_node
                succ_node.preds[0] = pred_node

                # append paddings to the end of the predecessors.
                pads_node = ir_node.preds[1]
                pads_node.op = "Pad"
                succ_node.preds.append(pads_node)

                self._ir_graph.nodes.remove(ir_node)
            else:
                continue

        return AxfcError.SUCCESS

    ## @internal
    #  This method is used to create a new IR node from tf.NodeDef and append it to the IR graph.
    #  The successors and predecessors of the IR node is found using the symbol table.
    #
    # @param self this object
    # @param tf_node_def input node_def object of Tensorflow
    # @return error info.
    def __append_node_def(self, tf_node_def: tf.compat.v1.NodeDef) -> AxfcError:
        # logging.info("AxfcTFIRBuilder:append_node_def - tf_node_def: %s", tf_node_def.name)

        # create a new IR node
        ir_node = AxfcIRNode(tf_node_def)

        # register the IR node to the symbol table with the name of node_def
        self._ir_symtab[tf_node_def.name] = ir_node

        # set the operation of this node
        ir_node.op = tf_node_def.op

        # set the name of this node by pruning unnecessary prefixes
        model_name = self._md.get_model_name()

        prefix_index = tf_node_def.name.find(model_name)
        if prefix_index >= 0:
            ir_node.name = tf_node_def.name[prefix_index:]
        else:
            ir_node.name = tf_node_def.name

        # check the node that is supported by AIXH hardware
        layer_info = self._md.get_layer_info(ir_node.op)

        if self._md.get_aixh_support(ir_node.op):
            ir_node.is_aixh_support = True
            ir_node.aixh_profit = layer_info.profit
        else:
            ir_node.is_aixh_support = False
            ir_node.aixh_profit = 0

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

    ## For debugging
    def __str__(self):
        # tf.io.write_graph(self._tf_graph, './', '../tst/graph_def.pbtxt', as_text=True)
        return
