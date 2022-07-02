

import enum
import multiprocessing
import subprocess
from multiprocessing import Process

from tensorflow.python.eager.context import remove_function
from util.AxfcTFGraphUtil import AxfcTFGraphUtil
from copy import deepcopy
import tensorflow as tf

class AxfcTFCustomGraph:

    ## @var __md
    # the machine description object

    ## @var __graph_def
    # the definition of the graph

    ## @var axfc_util 
    # util object to different framework library

    ## @var __path_module
    # the path to the custom operation module

    ## @var __output_type
    # output type of the graph

    ## @var __aix_graph_path
    # the path to the generated aix graph

    ## @var __ir_block_list = ir_blocks
    # list of the AxfcIRBlock computation object

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

    ## This method is used to load the custom operation then connecting it 
    #  to the tranposes for the inputs and outputs of the operation
    #
    #  @param self this object
    #  @param custom_graph the custom graph definition
    #  @param input_tensor_list the list of input tensors of the block
    #  @param aix_op_block_index the index number of the aix op in the entire model
    #  @param output_node_list the list of AxfcIRNode that represents the output of the block
    #  @param custom_op_path the path to the custom op module that has been generated
    #  @return custom_graph, aix_tensor refers to the tensor of AixOp, tensor_tranpose_NHWC_list which connected to the output of the AixOp
    def __load_aix_op(self, custom_graph, input_tensor_list, aix_op_block_index, output_node_list, custom_op_path):
        
        tf.compat.v1.disable_eager_execution()

        # launcher_path = "/home/hok/Documents/aix_pro/skt-aix-launcher"
        # custom_op_path = launcher_path + "/launcher/src/custom_op_kernel.so"
        # output_names = [node.name.replace("/","_").lower() for node in output_node_list]
        #Generate make file
        # sub_process = subprocess.run(f"rm {custom_op_path}", shell=True)
        # sub_process = subprocess.run(f"cd {launcher_path} && . venv/bin/activate && make generate_kernel OUT_NAMES={','.join(output_names)}", shell=True)
        # sub_process = subprocess.run(f"cp /home/hok/Documents/aix_pro/skt-aix-launcher/launcher/src/custom_op_kernel.so /home/hok/Documents/aix_pro/skt-aix-frontend-compiler/tst", shell=True)
        
        #Load custom kernel
        op_module = tf.load_op_library(custom_op_path)
        # op_module = tf.load_op_library(custom_op_path)
        
        #Connect tranpose to block's inputs
        with custom_graph.as_default() as custom_graph:
                tensor_transpose_list = []
                #convert the input node list into NCHW format for aix op 
                for input_tensor in input_tensor_list:
                    tensor_tranpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                    tensor_transpose_list.append(tensor_tranpose)

                list_id_str = ",".join([str(node.id) for node in output_node_list])

                #emit aix op
                aix_tensor = op_module.aix_op(
                    input = tensor_transpose_list,
                    output_type = self.__output_type,
                    aix_graph_path = self.__aix_graph_path + "%s" %aix_op_block_index,
                    output_ids = list_id_str
                )

        #Tranpose AixOp back to NHWC
        # with custom_graph.as_default() as custom_graph:
        #         aix_op_name = "AixOp:0" if aix_op_block_index == 0 else f"AixOp_{aix_op_block_index}:0"
        #         aix_op = custom_graph.get_tensor_by_name(aix_op_name)
        #         tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')                

        tensor_transpose_NHWC_list = []
        #Map output to transposes
        with custom_graph.as_default() as custom_graph:
            for index, _ in enumerate(output_node_list):
                aix_op_name = f"AixOp:{index}" if aix_op_block_index == 0 else f"AixOp_{aix_op_block_index}:{index}"
                aix_op = custom_graph.get_tensor_by_name(aix_op_name)
                tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')
                tensor_transpose_NHWC_list.append(tensor_transpose_NHWC)


        return custom_graph, aix_tensor, tensor_transpose_NHWC_list
    
    ## Detach all the nodes that supported by the Aix Graph (from the block) from the __graph_def in self
    #
    #  @param self this object
    #  @param ir_bock AxfcIRBlock type
    def detach_aix_graph_nodes(self, ir_block):
        
        #Clean ir_block nodes input such as padding and kernel
        # check_to_clean = []

        #For cleaning directly without checking
        # direct_clean = []
        
        #For cleaning input
        node_to_clean = []
        #detach nodes from ir_block
        graph_nodes = self.__graph_def.node 
        for node in ir_block.nodes:
            for index, node_def in enumerate(graph_nodes):
                if node_def.name == node.name:
                    del self.__graph_def.node[index]
                    break
            #_______________________
            node_to_clean += node.preds

        all_node_name = [node.name for node in self.__graph_def.node]
        keep_nodes = []
        for node in node_to_clean:
            #Skip Padding
            if node.op == "Pad":
                continue

            #check if current node is required from other node
            for succ_node in node.succs:
                if succ_node.name in all_node_name:
                    # node_to_clean.remove(node)
                    keep_nodes.append(node)
                    break
        
        node_to_clean = [node for node in node_to_clean if node not in keep_nodes]

        graph_nodes = self.__graph_def.node
        pad_to_clean = []
        for node in node_to_clean:
            #Remove Padding successors
            if node.op == "Pad":
                pad_to_clean += node.succs

            #Remove all Const, Identity and Pad
            if node.op in ["Const", "Identity", "Pad"]:
                for index, node_def in enumerate(graph_nodes):
                    if node.name == node_def.name:
                        del self.__graph_def.node[index]
                        break
        
        graph_nodes = self.__graph_def.node
        for pad_node in pad_to_clean:
            for index, node_def in enumerate(graph_nodes):
                if pad_node.name == node_def.name:
                    del self.__graph_def.node[index]
                    break
    
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
    
    ## For mapping aix tranpose_NHWC to connect to all of the block successors
    #  outside of the block after generating AixOp
    #
    #  @param output_node_list list of AxfcIRNode represent the output of th eblock
    #  @param tranpose_tensor_list tensors connected the AixOp outputs to the node in the model
    #  @param graph the graph to partitioning the output
    #  @return graph_def the partitioned graph
    def map_aix_tranpose_output(self, output_node_list, tranpose_tensor_list, graph):
        
        graph_def = graph.as_graph_def()
        
        for output_index, output_node in enumerate(output_node_list):
            last_node_name = output_node.name
            #Replace BiasaddClone for original FusedBatchNorm
            if "FusedBatchNorm" in last_node_name:
                last_node_name = last_node_name.split("/BiasaddClone")[0]

            for index, node in enumerate(graph_def.node):
                for input_index, input in enumerate(node.input):
                    if last_node_name == input:
                        graph_def.node[index].input[input_index] = tranpose_tensor_list[output_index].name.split(":")[0]
        
        return graph_def

    ## For removing Const and Identity node that has no connection
    #
    #  @param custom_graph the manipulated graph with AixOp tensor
    #  @return custom_graph that hsa Const and Identity nodes removed
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
    ## Compile all the aix supported block into AixOp
    #
    #  @param self this object
    #  @return custom_graph the completed custom graph with AixOps 
    def get_custom_graph(self):

        #Generate custom op outputs number based on largest number from aix graphs

        # launcher_path = "/home/hok/Documents/aix_pro/skt-aix-launcher"
        launcher_path = self.__path_module
        custom_op_path = launcher_path + "/launcher/src/custom_op_kernel.so"
        # output_names = [node.name.replace("/","_").lower() for node in output_node_list]
        output_number = max([len(ir_block.output_nodes) for ir_block in self.__ir_block_list])
        #Generate make file
        sub_process = subprocess.run(f"cd {launcher_path} && . venv/bin/activate && make generate_kernel OUT_NUM={output_number}", shell=True)

        self.__path_module = custom_op_path

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
            
            #output list id
            output_id_list = ir_block.output_nodes

            #get aixop graph
            aix_graph, aix_tensor, tensor_transpose_NHWC_list = self.__load_aix_op(custom_graph, input_tensor_list, index, output_id_list, custom_op_path)
            
            #map tensor_tranpose_NHWC to last_node outputs
            # last_node_name = ir_block.output_nodes[0].name
            # last_node_name = ir_block.output_node.name

            graph_def = self.map_aix_tranpose_output(ir_block.output_nodes, tensor_transpose_NHWC_list, aix_graph)
            
            #abrogate last_node_list input to tranpose in ir_block if ir_block is_aixh_support
            self.__graph_def = graph_def
            self.abrogate_node_succs(ir_block.output_nodes, tensor_transpose_NHWC_list)
            
            #detach the node in ir_block from custom graph
            self.detach_aix_graph_nodes(ir_block)
        
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(self.__graph_def, name="")
        
        return self.optimize_custom_graph(custom_graph)        

    ## Abrogate in the memory ir_node of the last aix ir graph's successor
    #  to connect to aix tranpose as input
    #
    #  @param abrogate_nodes the node to manipulate the input
    #  @param tensor_tranpose_NHWC_list list of the output of AixOp tensors
    def abrogate_node_succs(self, abrogate_nodes, tensor_transpose_NHWC_list):

        for node_index, abrogate_node in enumerate(abrogate_nodes):
            input_node_name = tensor_transpose_NHWC_list[node_index].name.split(":")[0]
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
                        block.input_nodes[index] = tensor_transpose_NHWC_list[node_index]