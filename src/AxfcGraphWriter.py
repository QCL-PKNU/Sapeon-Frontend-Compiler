#######################################################################
#   AxfcGraphUtil
#
#   Created: 2020. 08. 07
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import json
import logging

from AxfcError import *
from AxfcIRNode import *


#######################################################################
# AxfcGraphWriter class
#######################################################################
class AxfcGraphWriter:

    ## @var __edge_id
    # Edge's ID (auto increase)

    ## @var __graph
    # dictionary of edges and nodes

    ## @var __nodes
    # set of nodes

    ## @var __x_axis
    # x axis of edges

    ## @var __y_axis
    # y axis of edges

    ## The constructor
    def __init__(self):
        self.__edge_id = 0
        self.__graph = {'edges': [], 'nodes': []}
        self.__nodes = set()
        self.__x_axis = 0
        self.__y_axis = 0

    ## This method is used insert the edges of node
    #
    # @param self this object
    # @param source_node_id node's id for source 
    # @param target_node_id node's id for target
    def add_edge(self, source_node_id, target_node_id):

        # append to source and target id to __graph
        self.__graph["edges"].append({
            'id': self.__edge_id,
            'source': source_node_id,
            'target': target_node_id
        })

        self.__edge_id += 1

    ## This method is used insert the node
    #
    # @param self this object
    # @param ir_node AxfcIRNode node
    def add_node(self, ir_node: AxfcIRNode):

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

    ## This method is used to write the edges and nodes to Sigma js json format
    #
    # @param self this object
    # @param file_path file path for dumping the IR graph
    # @return error info
    def write_file(self, file_path: str) -> AxfcError:
        try:
            with open(file_path, 'w') as fd:
                json.dump(self.__graph, fd)
        except IOError as e:
            logging.warning("AxfcGraphWriter: dump_graph - %s", str(e))
            return AxfcError.DUMP_IR_GRAPH_ERROR

        return AxfcError.SUCCESS
