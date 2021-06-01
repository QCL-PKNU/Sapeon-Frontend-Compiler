#######################################################################
#   AxfcTFGraphUtil
#
#   Created: 2020. 09. 10
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from tensorflow.core.framework import graph_pb2
import tensorflow as tf 
import copy


#######################################################################
# AxfcTFGraphUtil class
#######################################################################

class AxfcTFGraphUtil:
    
    ## @var graph_def
    # frozen model GraphDef proto
    
    ## The constructor
    def __init__(self,graph_def):
        
        # check instance of graph_def
        if not isinstance(graph_def, graph_pb2.GraphDef):
            raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")
            
        self.graph_def  = graph_def
    
    ## This method is used to emit the sub graph by input and output node
    # 
    # @param self this object
    # @param orig_nodes the list of input node name
    # @param dest_nodes the list of output node name
    # @return graph_pb2.GraphDef the GraphDef proto object
    def extract_sub_graph(self, orig_nodes: list, dest_nodes: list)->graph_pb2.GraphDef:        
        
        # emit the nodes from begin
        sub_graph = self.extract_sub_graph_from_begin(self.graph_def, dest_nodes )
        
        # get the input tensor 
        tensor = self.get_tensor_by_name(orig_nodes[0])
        
        # create the sub-graph on sub_graph
        with tf.Graph().as_default() as graph:
            tensor = tf.compat.v1.placeholder(dtype=tensor.dtype, shape=tensor.shape,name=self.node_name(tensor.name))
            tf.import_graph_def(sub_graph, input_map={self.node_name(tensor.name): tensor}, name='')
        
        # take only the sub-graph from orig_nodes to dest_nodes
        sub_graph = self.extract_sub_graph_from_begin(graph.as_graph_def(), dest_nodes )

        return sub_graph
    
    ## This method i used to emit the sub graph by input and output node names
    # 
    # @param self this object
    # @param orig_nodes the list of input nodes
    # @param dest_nodes the list of output nodes
    # @return graph_pb2.GraphDef the GraphDef proto object
    def get_tensor_by_name(self, name):
        
        with tf.Graph().as_default() as _:
            tf.import_graph_def(self.graph_def, name='')
            tensor_input = tf.compat.v1.get_default_graph().get_tensor_by_name('{}:0'.format(name))
            return tensor_input
    
    ## This method is used to Extract the subgraph that can reach any of the nodes in 'dest_nodes'
    # 
    # @param self this object
    # @param graph_def A graph_pb2.GraphDef proto
    # @param dest_nodes A list of strings specifying the destination node names.
    # @return The GraphDef of the sub-graph.
    def extract_sub_graph_from_begin(self, graph_def, dest_nodes:list):

        name_to_input_name, name_to_node, name_to_seq_num = self._extract_graph_summary(graph_def)
        
        # check if the destination nodes exist in graph
        self._assert_nodes_are_present(name_to_node, dest_nodes)
        
        # emit the reachable nodes
        nodes_to_keep = self._bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
    
        nodes_to_keep_list = sorted(
            list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
        
        # Now construct the output GraphDef
        out = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
            out.node.extend([copy.deepcopy(name_to_node[n])])
        out.library.CopyFrom(graph_def.library)
        out.versions.CopyFrom(graph_def.versions)
    
        return out

    ## this method is used to search for reachable nodes from target nodes
    # 
    # @param self this object
    # @param target_nodes the list of destination node names
    # @param name_to_input_name the dictionary of name with input name
    # @return the nodes that are reachable
    def _bfs_for_reachable_nodes(self, target_nodes, name_to_input_name):

        nodes_to_keep = set()
        
        # Breadth first search to find all the nodes that we should keep.            
        next_to_visit = target_nodes[:]
        while next_to_visit:
            node = next_to_visit[0]
            del next_to_visit[0]
            if node in nodes_to_keep:
                # Already visited this node.
                continue
            nodes_to_keep.add(node)
            if node in name_to_input_name:
                next_to_visit += name_to_input_name[node]
                
        return nodes_to_keep
    
    ## Assert that nodes are present in the graph.
    # 
    # @param self this object
    # @param name_to_node the dictionary name to spacific node
    # @param nodes the list of node names
    def _assert_nodes_are_present(self, name_to_node, nodes):

        for d in nodes:
            assert d in name_to_node, "%s is not in graph" % d
    
    ## this method is used to validate the name of node
    # 
    # @param self this object
    # @param name the name of node 
    # @return the valid name node
    def node_name(self,name):
        if name.startswith("^"):
            return name[1:]
        else:
            return name.split(":")[0]
    
    ## this method is used to extracts useful information from the graph and returns them.
    # 
    # @param self this object
    # @param graph_def the frozen model GraphDef proto
    # @return name_to_input_name, name_to_node, name_to_seq_num 
    def _extract_graph_summary(self,graph_def):

        name_to_input_name = {}  # Keyed by the dest node name.
        name_to_node = {}  # Keyed by node name.
    
        # Keeps track of node sequences. It is important to still output the
        # operations in the original order.
        name_to_seq_num = {}  # Keyed by node name.
        seq = 0
        for node in graph_def.node:
            n = self.node_name(node.name)
            name_to_node[n] = node
            name_to_input_name[n] = [self.node_name(x) for x in node.input]
            # Prevent colocated nodes from being lost.
            if "_class" in node.attr:
                for colocated_node_name in node.attr["_class"].list.s:
                    name_to_input_name[n].append(
                        self._get_colocated_node_name(colocated_node_name))
            name_to_seq_num[n] = seq
            seq += 1
        return name_to_input_name, name_to_node, name_to_seq_num
    
    def _get_colocated_node_name(self, colocated_node_name):
        """Decodes colocated node name and returns it without loc:@ prepended."""
        colocated_node_decoded = colocated_node_name.decode("utf-8")
        if colocated_node_decoded.startswith("loc:@"):
            return colocated_node_decoded[5:]
        return colocated_node_decoded