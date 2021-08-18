

import enum
import multiprocessing
from multiprocessing import Process

from tensorflow.python.eager.context import remove_function
from util.AxfcTFGraphUtil import AxfcTFGraphUtil
from copy import deepcopy
import tensorflow as tf

class AxfcCustomGraphV2:
    def __init__(self, graph_def,
                path_module,
                output_type,
                ir_blocks,
                aix_graph_path,
                input_tensors,
                output_tensors,
                md):
        self.__md = md
        self.__graph_def = graph_def
        self.__axfc_util = AxfcTFGraphUtil(graph_def)
        self.__path_module = path_module
        self.__output_type = output_type
        self.__aix_graph_path = aix_graph_path
        self.__ir_block_list = ir_blocks

    #For connecting aix op and tranposes to the input and output of the block
    def __load_aix_op(self, custom_graph, input_tensor_list, aix_op_block_index):
        
        tf.compat.v1.disable_eager_execution()
        op_module = tf.load_op_library(self.__path_module)

        #Connect tranpose to block's inputs
        with custom_graph.as_default() as custom_graph:
                tensor_transpose_list = []
                #convert the input node list into NCHW format for aix op 
                for input_tensor in input_tensor_list:
                    tensor_tranpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                    tensor_transpose_list.append(tensor_tranpose)
                
                #emit aix op
                aix_tensor = op_module.aix_op(
                    input = tensor_transpose_list,
                    output_type = self.__output_type,
                    aix_graph_path = self.__aix_graph_path + "%s" %aix_op_block_index
                )
        #Tranpose AixOp back to NHWC
        with custom_graph.as_default() as custom_graph:
                aix_op_name = "AixOp:0" if aix_op_block_index == 0 else f"AixOp_{aix_op_block_index}:0"
                aix_op = custom_graph.get_tensor_by_name(aix_op_name)
                tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')                

        return custom_graph, aix_tensor, tensor_transpose_NHWC
    
    #Detach all the nodes that supported by the AIX graph from __graph_def
    def detach_aix_graph_nodes(self, ir_block, ignore_nodes):
        
        #Clean ir_block nodes input such as padding and kernel
        input_to_clean = []
        
        #detach nodes from ir_block
        graph_nodes = self.__graph_def.node 
        for node in ir_block.nodes:
            for index, node_def in enumerate(graph_nodes):
                if node_def.name == node.name:
                    del self.__graph_def.node[index]
                    break

            for input in node.node_def.input:
                input_to_clean.append(input)

        graph_nodes = self.__graph_def.node
        manager = multiprocessing.Manager()
        
        process_list = []
        return_dict = manager.dict() #For storing that node that can be removed
        #Use multiprocess to check whether or not the node can be removed
        for input_node in input_to_clean:
            p = Process(target=self.check_input_node, args=(input_node, graph_nodes, return_dict,))
            process_list.append(p)
            p.start()

        for p in process_list:
            p.join()
        
        #Remove all the nodes from return_dict
        for node in return_dict.values():
            self.__graph_def.node.remove(node)
    
    #This function is used to check the whether or not the input node can be remove
    #It will check if the input node is required as input or has connection to another node
    def check_input_node(self, input_node, graph_nodes, return_dict):

        for index, node_def in enumerate(graph_nodes):
                
            if input_node == node_def.name and node_def.op in ["Const", "Identity", "Pad"]:
                #Need to check if other node requires the Const or Identity as input
                is_required = False
                for node in self.__graph_def.node:
                    if input_node in node.input:
                        is_required = True
                        break
                
                #If no node requires it as input then return it in the dictionary
                if not is_required:
                    return_dict[node_def.name] = self.__graph_def.node[index]
                
                break
    
    #For maping aix tranpose_NHWC to connect to all of the block successors
    #outside of the block
    def map_aix_tranpose_output(self, last_node_name, tranpose_tensor, graph):
        
        graph_def = graph.as_graph_def()
        for index, node in enumerate(graph_def.node):
            for input_index, input in enumerate(node.input):
                if last_node_name == input:
                    graph_def.node[index].input[input_index] = tranpose_tensor.name.split(":")[0]
                    
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(graph_def, name="")
            
        return custom_graph

    #For removing Const and Identity node that has no connection
    def optimize_custom_graph(self, custom_graph):
        
        custom_graph_def = custom_graph.as_graph_def()
        # take only tensor that has connection
        node_to_remove = []
        for index, op in enumerate(custom_graph.get_operations()):
            if not op.inputs and not op.outputs[0].consumers() and op.type in ["Const", "Identity"]:
                node_to_remove.append(custom_graph_def.node[index])
        
        #remove node
        for node in node_to_remove:
            custom_graph_def.node.remove(node)
        
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(custom_graph_def, name="")
        
        return custom_graph
    
    #Compile all the aix supported block into AixOp
    def get_custom_graph(self):
        
        for index, ir_block in enumerate(self.__ir_block_list):
            #Take input list from the first node in the block
            
            # input_node_list = ir_block.nodes[0].node_def.input
            input_node_list = [node.name for node in ir_block.input_nodes if node.op != "Const"]
            
            with tf.Graph().as_default() as custom_graph:
                tf.import_graph_def(self.__graph_def, name="")
            
            #Get input_tensor_list
            input_tensor_list = []
            for input_node in ir_block.input_nodes:
                #Ignore Const, Pad and Identity node
                if input_node.op in ["Const", "Pad", "Identity"] and len(ir_block.input_nodes) > 1:
                    continue
                
                #validate input name, in case it is a tensor from previous aixop
                input_name = input_node.name.split(":")[0]
                
                #get the gensor opeator for all the input node list except /kernel
                tensor = custom_graph.get_tensor_by_name("{}:0".format(input_name))
                if tensor not in input_tensor_list:
                    input_tensor_list.append(tensor)
            
            #get aixop graph
            aix_graph, aix_tensor, tensor_transpose_NHWC = self.__load_aix_op(custom_graph, input_tensor_list, index)
            
            #map tensor_tranpose_NHWC to last_node outputs
            last_node_name = ir_block.output_node.name
            custom_graph = self.map_aix_tranpose_output(last_node_name, tensor_transpose_NHWC, aix_graph)
            
            #abrogate last_node_list input to tranpose in ir_block if ir_block is_aixh_support
            self.abrogate_node_succs(ir_block.output_node, tensor_transpose_NHWC)
            
            #detach the node in ir_block from custom graph
            self.__graph_def = custom_graph.as_graph_def()
            self.detach_aix_graph_nodes(ir_block, input_node_list)
        
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(self.__graph_def, name="")
        
        return self.optimize_custom_graph(custom_graph)        

    #Abrogate in memory ir_node of last aix ir graph node's successor to connect to aix tranpose as input
    def abrogate_node_succs(self, abrogate_node, input_node):
        input_node_name = input_node.name.split(":")[0]
        #Change the input of the node that connected to the 
        #last node that have been turned into AixOp
        for abrogate_node_succs in abrogate_node.succs:
            for index, succ_node_input in enumerate(abrogate_node_succs.node_def.input):
                if succ_node_input == abrogate_node.name:
                    abrogate_node_succs.node_def.input[index] = input_node_name
        
        #Change the input of the block if it is contains
        #the node that have been turned into AixOp
        for block in self.__ir_block_list:
            for index, block_input_node in enumerate(block.input_nodes):
                if abrogate_node == block_input_node:
                     block.input_nodes[index] = input_node