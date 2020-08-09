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

import webview
import json

#######################################################################
# AxfcGraphUtil class
#######################################################################
class AxfcGraphUtil:

    ## @var __edge_id
    # Edge's ID (auto increase)

    ## @var _edges_nodes
    # dictionary of edges and nodes

    ## @var _nodes
    # set of nodes

    ## @var __x_axis
    # x axis of edges

    ## @var __y_axis
    # y axis of edges

    ## The constructor
    def __init__(self):
        self.__edge_id = 0
        self._edges_nodes = {'edges':[],'nodes':[] }
        self._nodes = set()
        self.__x_axis = 0
        self.__y_axis = 0

    ## This method is used insert the edges of node
    #
    # @param self this object
    # @param source_node_id node's id for source 
    # @param target_node_id node's id for target
    def add_edge(self, source_node_id, target_node_id):

        # append to source and target id to _edges_nodes
        self._edges_nodes["edges"].append({
            'id': self.__edge_id,
            'source': source_node_id,
            'target': target_node_id
        })

        self.__edge_id += 1

    ## This method is used insert the node
    #
    # @param self this object
    # @param ir_node AxfcIRNode node
    # @param x horizontal x-axis when display on graph
    def add_node(self, ir_node):
            
        # check if new node is not dubplicate
        # then add this node into _edges_nodes 
        if (ir_node not in self._nodes):
            self._edges_nodes['nodes'].append({
                'id': ir_node.id,
                'label': ir_node.op,
                'x': self.__x_axis,
                'y': self.__y_axis,
                'size':1,
                'attributes': {
                    'is_root': ir_node.is_root,
                    'is_aixh_support': ir_node.is_aixh_support,
                    'name': ir_node.node_def.name,
                    'op': ir_node.op
                }
            })

            self._nodes.add(ir_node)

    ## This method is used to write the edges and nodes to Sigma js json format
    #
    # @param self this object      
    def build(self):
        with open('src/axfc_core_display/axfc_data.json','w') as file:
            json.dump(self._edges_nodes, file)

    ## This method is used to display the graph by using pywebview
    #
    # @param self this object   
    def show(self):
        webview.create_window('Axfc Graph', '../src/axfc_core_display/display.html')
        webview.start(http_server=True)
