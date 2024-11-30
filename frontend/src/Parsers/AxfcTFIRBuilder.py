#######################################################################
#   AxfcTFIRBuilder
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################

import tensorflow as tf

from . import AxfcIRBuilder
from AxfcGraphWriter import *


#######################################################################
# AxfcTFIRBuilder class
#######################################################################

class AxfcTFIRBuilder(AxfcIRBuilder):

    ## The constructor
    def __init__(self, md):
        super().__init__(md)

        self.__tf_graph = None


    def _read_model_graph(self, path: str) -> AxfcError:
        """
        Reads the TensorFlow graph from the provided model path.

        Args:
            path (str): Path to the saved TensorFlow model.
        """
        logging.info("AxfcTFIRBuilder:read_model_graph - path: %s", path)

        graph = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(path, 'rb') as f:
            graph.ParseFromString(f.read())
            tf.import_graph_def(graph)

        # Remove identity nodes
        self.__tf_graph = tf.compat.v1.graph_util.remove_training_nodes(graph, protected_nodes=None)

        return AxfcError.SUCCESS
    

    def _build_naive_ir(self, path: str) -> AxfcError:
        """
        Builds a naive IR from the TensorFlow graph.

        Args:
            path (str): Path to the TensorFlow model.
        """
        logging.info("AxfcTFIRBuilder:build_naive_ir - path: %s", path)

        err = self._read_model_graph(path)
        if err !=  AxfcError.SUCCESS:
            return err

        graph_def = self.__tf_graph
        if graph_def is None:
            return AxfcError.INVALID_TF_GRAPH
        
        # Add all graph_def nodes into the symbolic IR table
        for node_def in graph_def.node:
            err = self.__append_node_sym_ir(node_def)

        # Build AIX IR Graph using TensorFlow graph nodes
        for node_def in graph_def.node:
            err = self.__append_node_def(node_def)
            if err != AxfcError.SUCCESS:
                return err

        # Remove unecessary IR Nodes
        return self.__prune_ir_nodes()


    def __prune_ir_nodes(self) -> AxfcError:
        """Prunes unnecessary nodes (e.g., Identity and Pad) from the IR graph."""

        for ir_node in self._ir_graph.nodes:
            # # Handle Identity nodes
            # if ir_node.op == "Identity":
            #     print("----------")
            #     if len(ir_node.preds) != 1 or len(ir_node.succs) != 1:
            #         continue

            #     # Get the predecessor and successor nodes
            #     pred_node = ir_node.preds[0]
            #     succ_node = ir_node.succs[0]

            #     # Transfer the tensor value from Identity to the successor (key layer)
            #     if "input_tensor_values" not in succ_node.attrs:
            #         succ_node.attrs["input_tensor_values"] = {}
            #     succ_node.attrs["input_tensor_values"][ir_node.name] = pred_node.tensor_value

            #     # Rewire the graph: connect predecessor directly to successor
            #     pred_node.succs[0] = succ_node
            #     for i, node in enumerate(succ_node.preds):
            #         if node == ir_node:
            #             succ_node.preds[i] = pred_node
            #             break

            #     # Remove the Identity node from the IR graph
            #     self._ir_graph.nodes.remove(ir_node)

            # Handle Pad nodes
            if ir_node.op == "Pad":
                if len(ir_node.preds) != 2 or len(ir_node.succs) != 1:
                    continue  # Skip invalid Pad nodes

                # Check the following nodes
                succ_node: AxfcIRNode = ir_node.succs[0]
                if succ_node.op not in ["Conv2D", "DepthwiseConv2dNative", "MaxPool", "AvgPool"]:
                    continue # Skip Pad nodes not followed by a supported operation

                # Remove the Pad node and connect its input directly
                pred_node: AxfcIRNode = ir_node.preds[0]
                pads_node: AxfcIRNode = ir_node.preds[1]

                # Rewire the graph: connect predecessor directly to successor
                pred_node.succs = [succ_node]
                for i, node in enumerate(succ_node.preds):
                    if node == ir_node:
                        succ_node.preds[i] = pred_node
                        break

                # Remove the Pad node from the IR graph
                self._ir_graph.nodes.remove(ir_node)

        return AxfcError.SUCCESS


    def __append_node_sym_ir(self, node_def: tf.compat.v1.NodeDef) -> AxfcError:
        """
        Adds a TensorFlow node to the symbolic IR table.

        Args:
            node_def: TensorFlow NodeDef object representing a graph node.
        """
        ir_node = AxfcIRNode(node_def)
        self._ir_symtab[node_def.name] = ir_node

        return AxfcError.SUCCESS


    def __append_node_def(self, node_def: tf.compat.v1.NodeDef) -> AxfcError:
        """
        Creates an IR node from a TensorFlow node and appends it to the IR graph.

        Args:
            node_def: TensorFlow NodeDef object representing a graph node.
        """
        ir_node: AxfcIRNode = self._ir_symtab.get(node_def.name)
        ir_node.op = node_def.op
        ir_node.name = node_def.name

        # Check if the node is supported by target hardware
        layer_info = self._md.get_layer_info(ir_node.op)
        if self._md.get_aixh_support(ir_node.op) and not self._md.BREAK_POINT_CONDITION:
            ir_node.is_aixh_support = True
            ir_node.aixh_profit = layer_info.profit
        else:
            ir_node.is_aixh_support = False
            ir_node.aixh_profit = 0

        # Connect predecessor and successors
        for pred_name in node_def.input:
            pred_name = pred_name.split(":")[0].replace("^", "")

            # Find the predecessor using the symbol table
            pred_node: AxfcIRNode = self._ir_symtab.get(pred_name)
            if pred_node:
                pred_node.succs.append(ir_node)
                ir_node.preds.append(pred_node)
            else:
                return AxfcError.PRED_NODE_NOT_FOUND
            

        # Append the new IR node to the graph
        self._ir_graph.append_node(ir_node)

        # Handle breakpoint nodes
        if ir_node.name == self._md.get_break_point_node():
            self._md.BREAK_POINT_CONDITION = True

        return AxfcError.SUCCESS


    ## For debugging
    def __str__(self):
        # tf.io.write_graph(self.__tf_graph, './', '../tst/graph_def.pbtxt', as_text=True)
        return
