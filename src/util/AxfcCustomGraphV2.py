

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
                output_tensors):
        
        self.__graph_def = graph_def
        self.__axfc_util = AxfcTFGraphUtil(graph_def)
        self.__path_module = path_module
        self.__output_type = output_type
        self.__aix_graph_path = aix_graph_path
        self.__ir_block_list = ir_blocks
        self.__input_tensors = input_tensors
        self.__output_tensors = output_tensors
        self.__input_node_to_check = ["input_1_1"] #to check in optimization

    def get_custom_graph(self):
        
        tf.compat.v1.disable_eager_execution()
        op_module = tf.load_op_library(self.__path_module)

        #Loop through support ir_block supported by aix graph
        for index, ir_block in enumerate(self.__ir_block_list):
            #Take input list from the first node in the block
            input_node_list = ir_block.nodes[0].node_def.input

            #extract sub_graph from begin until the input_node_list
            sub_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, input_node_list)

            with tf.Graph().as_default() as custom_graph:
                tf.import_graph_def(sub_graph, name="")

            #Get input_tensor_list
            input_tensor_list = []
            for input_node in input_node_list:
                #ignore kernel node
                if "/kernel" in input_node:
                    continue
                #get the gensor opeator for all the input node list except /kernel
                tensor = custom_graph.get_tensor_by_name("{}:0".format(input_node))
                input_tensor_list.append(tensor)
            
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
                    aix_graph_path = self.__aix_graph_path + "%s" %index
                )
            #Get the last node from the supported ir block
            last_node_list = [ir_block.nodes[-1].name]
            #Get the last outputs of the graph
            output_node_list = [i.name for i in self.__output_tensors]
            
            #save node from the custom graph for optimization purpose
            optimized_node_list = [node.name for node in custom_graph.as_graph_def().node]

            #extract sub graph from model start from the last_node_list until the output_node_list
            tensor_def = self.__axfc_util.extract_sub_graph(last_node_list, output_node_list)
            
            #merge aixop grpah with sub graph
            with custom_graph.as_default() as custom_graph:
                
                aix_op_name = "AixOp:0" if index == 0 else f"AixOp_{index}:0"

                aix_op = custom_graph.get_tensor_by_name(aix_op_name)
                tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')

                tf.import_graph_def(tensor_def,
                                    input_map={last_node_list[0]: tensor_transpose_NHWC},
                                    name="")
            
            #reset the graph def for next aix 
            self.__axfc_util.graph_def = custom_graph.as_graph_def()
            self.__graph_def = custom_graph.as_graph_def()

            node_to_check = []
            for optimize_node in optimized_node_list:
                if "/kernel" not in optimize_node and "Transpose_" not in optimize_node and "AixOp" not in optimize_node and "Pad" not in optimize_node:
                    node_to_check.append(optimize_node)
                
            custom_graph = self.optimize_graph_node(custom_graph, node_to_check, input_node_list, last_node_list)

            if index == 2:
                break

        return custom_graph

    def optimize_graph_node(self, custom_graph, node_to_check, input_node_list = [], last_node_list= []):
        
        custom_graph_def = custom_graph.as_graph_def()
        node_to_remove = []
        
        for input_node in input_node_list:
            if "/kernel" in input_node:
                #Ignore /kernel node
                continue
                try:
                    node_tensor = custom_graph.get_tensor_by_name(f"{input_node}:0")
                    node_to_remove.append(node_tensor.op.name)
                except Exception as e:
                    continue
            else:
                self.__input_node_to_check += [input_node+"_1"]
        
        #Search to find the node from last to first input
        node_connection = [node for node in self.__input_node_to_check]
        while(node_connection):
            input_node = node_connection.pop()
            try:
                node_tensor = custom_graph.get_tensor_by_name(f"{input_node}:0")
                node_to_remove.append(node_tensor.op.name)

                for node_input in node_tensor.op.inputs:
                    node_connection.append(node_input.op.name)
            except Exception as e:
                continue

        #Connecting node to its original input
        for check_node in node_to_check:
            for index, node in enumerate(custom_graph_def.node):
                for input_index, input in enumerate(node.input):
                    if check_node in input and check_node != input:
                        custom_graph_def.node[index].input[input_index] = check_node
        
        #Clean duplicate
        node_to_remove = set(node_to_remove)
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(custom_graph_def, name='')
        custom_graph_def = custom_graph.as_graph_def()

        #Append the remove node to the list
        remove_node_def_list = []
        for index, op in enumerate(custom_graph.get_operations()):
            
            #remove all last_node_list that exist in custom graph
            for node_name in last_node_list:
                if op.name == node_name:
                    remove_node_def_list.append(custom_graph_def.node[index])
            #Remove the node from extracting graph
            for node_name in node_to_remove:
                if op.name == node_name:
                    remove_node_def_list.append(custom_graph_def.node[index])
        
        #Start remove the node from graph def
        for remove_node_def in remove_node_def_list:
            custom_graph_def.node.remove(remove_node_def)
        
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(custom_graph_def, name='')
        custom_graph_def = custom_graph.as_graph_def()
        
        # take only tensor that has connection
        for index, op in enumerate(custom_graph.get_operations()):
            if not op.inputs and not op.outputs[0].consumers():
                del custom_graph_def.node[index]
                break
        
        with tf.Graph().as_default() as main_graph:
            tf.import_graph_def(custom_graph_def, name='')
        
        return main_graph

    def __load_aix_op(self, input_node_list, output_node_list, aix_graph_path):

        input_tensor_list, sub_graph = self.__emit_input_tensors(input_node_list, output_node_list)

        tf.compat.v1.disable_eager_execution()

        op_module = tf.load_op_library(self.__path_module)

        with sub_graph.as_default() as aix_graph:
            
            tensor_transpose_list = []
            for input_tensor in input_tensor_list:

                tensor_transpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                tensor_transpose_list.append(tensor_transpose)

            aix_tensor = op_module.aix_op(
                input = tensor_transpose_list,
                output_type = self.__output_type,
                aix_graph_path = aix_graph_path
            )

        return aix_tensor, aix_graph

    def __emit_input_tensors(self, input_node_list, output_node_list):

        sub_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, input_node_list, output_node_list)

        # sub_graph = self.__axfc_util.extract_sub_graph(input_node_list, output_node_list)

        with tf.Graph().as_default() as import_graph:
            tf.import_graph_def(sub_graph, name="")
        
        input_tensor_list = []

        for input_node in input_node_list:
            #ignore conv1/kernel
            # if input_node == "conv1/kernel":
            #     continue
            tensor = import_graph.get_tensor_by_name("{}:0".format(input_node))
            input_tensor_list.append(tensor)

        return input_tensor_list, import_graph