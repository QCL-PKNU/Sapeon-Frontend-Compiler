#######################################################################
#   AxfcGraphUtil
#
#   Created: 2020. 08. 07
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Quantum Computing Labaratory (qcl.pknu.ac.kr)
#######################################################################

import json
import logging

from AxfcError import *
from AxfcIRNode import *


#######################################################################
# AxfcGraphWriter class
#######################################################################
class AxfcGraphWriter:

    """
    Manages the construction and representation of a graph, tracking nodes, edges, 
    and their respective positions.

    Attributes:
        __edge_id (int): Auto-incrementing identifier for each edge.
        __graph (dict): Stores 'edges' and 'nodes' in separate lists.
        __nodes (set): A set of unique nodes in the graph.
        __x_axis (int): Tracks the x-axis position for node placement.
        __y_axis (int): Tracks the y-axis position for node placement.
    """

    def __init__(self):
        self.__edge_id = 0
        self.__graph = {'edges': [], 'nodes': []}
        self.__nodes = set()
        self.__x_axis = 0
        self.__y_axis = 0


    def add_edge(self, source_node_id, target_node_id):
        """Inserts an edge between two nodes in the graph.

        Args:
            source_node_id: The ID of the source node.
            target_node_id: The ID of the target node.
        """
        self.__graph["edges"].append({
            'id': self.__edge_id,
            'source': source_node_id,
            'target': target_node_id
        })
        self.__edge_id += 1


    def add_node(self, ir_node: AxfcIRNode):
        """Inserts a node into the graph if it's not already present.

        Args:
            ir_node: An instance of AxfcIRNode to be added to the graph.
        """

        # check if new node is not dubplicate
        # then add this node into __graph
        if ir_node not in self.__nodes:
            self.__graph['nodes'].append({
                'id': ir_node.id,
                'label': ir_node.op,
                'x': self.__x_axis,
                'y': self.__y_axis,
                'size': 1,
                'attributes': {
                    'block_id': (ir_node.block_ref.id if ir_node.block_ref is not None else None),
                    'profit': ir_node.aixh_profit,
                    'is_aixh_support': ir_node.is_aixh_support,
                    'name': ir_node.node_def.name,
                    'op': ir_node.op
                }
            })

            self.__nodes.add(ir_node)


    def write_file(self, file_path: str) -> AxfcError:
        """Writes the edges and nodes to a file in Sigma.js JSON format.

        Args:
            file_path: The path to the file where the IR graph should be dumped.

        Returns:
            AxfcError: Error code indicating the success or failure of the operation.
        """
        try:
            with open(file_path, 'w') as fd:
                json.dump(self.__graph, fd, indent=4)
        except IOError as e:
            logging.warning("AxfcGraphWriter: dump_graph - %s", str(e))
            return AxfcError.DUMP_IR_GRAPH_ERROR
        
        logging.info("IR graph successfully dumped to %s", file_path)
        return AxfcError.SUCCESS
