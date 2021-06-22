

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

    def get_custom_graph(self):
        
        tf.compat.v1.disable_eager_execution()
        op_module = tf.load_op_library(self.__path_module)

        for index, ir_block in enumerate(self.__ir_block_list):
            input_node_list = ir_block.nodes[0].node_def.input

            sub_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, input_node_list)

            with tf.Graph().as_default() as custom_graph:
                tf.import_graph_def(sub_graph, name="")

            #Get input_tensor_list
            input_tensor_list = []
            for input_node in input_node_list:
                #ignore kernel node
                if "/kernel" in input_node:
                    continue

                tensor = custom_graph.get_tensor_by_name("{}:0".format(input_node))
                input_tensor_list.append(tensor)
            
            with custom_graph.as_default() as custom_graph:
                tensor_transpose_list = []

                for input_tensor in input_tensor_list:
                    tensor_tranpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                    tensor_transpose_list.append(tensor_tranpose)
                
                aix_tensor = op_module.aix_op(
                    input = tensor_transpose_list,
                    output_type = self.__output_type,
                    aix_graph_path = self.__aix_graph_path + "%s" %index
                )
            
            last_node_list = [ir_block.nodes[-1].name]

            #Test to the last node
            output_node_list = [i.name for i in self.__output_tensors]

            optimized_node_list = [node.name for node in custom_graph.as_graph_def().node]

            tensor_def = self.__axfc_util.extract_sub_graph(last_node_list, output_node_list)
            
            with custom_graph.as_default() as custom_graph:
                
                aix_op_name = "AixOp:0" if index == 0 else f"AixOp_{index}:0"

                aix_op = custom_graph.get_tensor_by_name(aix_op_name)
                tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')

                tf.import_graph_def(tensor_def,
                                    input_map={last_node_list[0]: tensor_transpose_NHWC},
                                    name="")
            
            #Test
            # if index == 2:
            #     return custom_graph
            
            self.__axfc_util.graph_def = custom_graph.as_graph_def()
            self.__graph_def = custom_graph.as_graph_def()

            node_to_check = []
            for optimize_node in optimized_node_list:
                if "/kernel" not in optimize_node and "Transpose_" not in optimize_node and "AixOp" not in optimize_node and "Pad" not in optimize_node:
                    node_to_check.append(optimize_node)
                    print (optimize_node, "appended")

            custom_graph = self.optimize_graph_node(custom_graph, node_to_check)

            if index == 2:
                return custom_graph

        return custom_graph

    def optimize_graph_node(self, custom_graph, node_to_check):
        
        custom_graph_def = custom_graph.as_graph_def()

        # print (node_to_check)
        # #Ignore some nodes 
        # node_to_ignore = ["/kernel", "Transpose_", "AixOp", "Pad"]
        # cleaned_node_to_check = node_to_check
        # for ignore_node in node_to_ignore:
        #     for node in node_to_check:
        #         if ignore_node in node:
        #             # cleaned_node_to_check.remove(node)
        #             print(node)

        for check_node in node_to_check:
            for index, node in enumerate(custom_graph_def.node):
                for input_index, input in enumerate(node.input):
                    if check_node in input and check_node != input:
                        custom_graph_def.node[index].input[input_index] = check_node
        
        # take only tensor that has connection
        for index, op in enumerate(custom_graph.get_operations()):
            if not op.inputs and not op.outputs[0].consumers():
                del custom_graph_def.node[index]
            #     break

        #directly override the node with predecessor to first node
        # for index, node in enumerate(custom_graph_def.node):
        #     if "input_1_1" in node.input:
        #         custom_graph_def.node[index].input[0] = "input_1"
            
        #     if "res3d_relu/Relu_1" in node.input:
        #         custom_graph_def.node[index].input[0] = "res3d_relu/Relu"
            
        #     if node.name == "P4_upsampled/Shape" or node.name == "C3_reduced/convolution":
        #         custom_graph_def.node[index].input[0] = "res3d_relu/Relu"
        
        with tf.Graph().as_default() as main_graph:
            tf.import_graph_def(custom_graph_def, name='')
        
        return main_graph

    def get_custom_graph_old(self):
        
        #First Block
        input_node_list = self.__ir_block_list[0].nodes[0].node_def.input

        sub_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, input_node_list)
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(sub_graph, name="")

        input_tensor_list = []

        for input_node in input_node_list:
            # ignore conv1/kernel
            if input_node == "conv1/kernel":
                continue
                
            tensor = custom_graph.get_tensor_by_name("{}:0".format(input_node))
            input_tensor_list.append(tensor)
            
        tf.compat.v1.disable_eager_execution()

        op_module = tf.load_op_library(self.__path_module)

        with custom_graph.as_default() as custom_graph:
            
            tensor_transpose_list = []
            for input_tensor in input_tensor_list:

                tensor_transpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                tensor_transpose_list.append(tensor_transpose)

            aix_tensor = op_module.aix_op(
                input = tensor_transpose_list,
                output_type = self.__output_type,
                aix_graph_path = self.__aix_graph_path
            )
        
        last_node_list = [self.__ir_block_list[0].nodes[-1].name] # -> Should be a list???
        # output_node_list = ["res3d_relu/Relu"]
        
        #Test
        output_node_list = [i.name for i in self.__output_tensors]

        tensor_def = self.__axfc_util.extract_sub_graph(last_node_list, output_node_list)

        with custom_graph.as_default() as custom_graph:

            aix_op = custom_graph.get_tensor_by_name('AixOp:0')
            tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')

            tf.import_graph_def(tensor_def,
                                input_map={last_node_list[0]: tensor_transpose_NHWC},
                                name="")
        
        self.__axfc_util.graph_def = custom_graph.as_graph_def()
        self.__graph_def = custom_graph.as_graph_def()

        #continue to second block
        input_node_list = self.__ir_block_list[1].nodes[0].node_def.input

        sub_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, input_node_list)
        
        with tf.Graph().as_default() as custom_graph:
            tf.import_graph_def(sub_graph, name="")

        input_tensor_list = []

        for input_node in input_node_list:
            # ignore conv1/kernel
            if "/kernel" in input_node:
                continue
                
            tensor = custom_graph.get_tensor_by_name("{}:0".format(input_node))
            input_tensor_list.append(tensor)

        # transpose_input = custom_graph.get_tensor_by_name("res3d_relu/Relu:0")
        
        with custom_graph.as_default() as custom_graph:
        # with tf.Graph().as_default() as custom_graph:


            # tensor_transpose = tf.transpose(transpose_input, [0, 3, 1, 2], name='Transpose_to_NCHW')

            tensor_transpose_list = []
            for input_tensor in input_tensor_list:

                tensor_transpose = tf.transpose(input_tensor, [0, 3, 1, 2], name='Transpose_to_NCHW')
                tensor_transpose_list.append(tensor_transpose)

            graph_path = "/home/hok/Documents/aix/skt-aix-frontend-compiler/tst/aix_graph.out.01"
            
            aix_tensor = op_module.aix_op(
                input = tensor_transpose_list,
                output_type = self.__output_type,
                aix_graph_path = graph_path
            )
        #Testing return custom_grpah
        # return custom_graph
        
        last_node_list = [self.__ir_block_list[1].nodes[-1].name, "P4_upsampled/Shape", "C3_reduced/convolution"] # -> Should be a list???
        # output_node_list = ["res4f_relu/Relu"]

        # node_to_skip_list = [i.name for i in custom_graph.as_graph_def().node if i.name not in input_node_list]

        node_to_skip_list = [i.name for i in custom_graph.as_graph_def().node if i.name != "res4a_branch2a/kernel"]

        node_to_skip_list = []
        
        tensor_def = self.__axfc_util.extract_sub_graph(last_node_list, output_node_list, node_to_skip_list)

        print (node_to_skip_list, "node to skip list")

        print("finished extract sub graph")

        f = open("tensor_def_node.txt", "w")
        for node in tensor_def.node:
            f.write(node.name + "\n")
        f.close()

        #Test return custom_graph
        # return custom_graph

        with custom_graph.as_default() as custom_graph:
            aix_op = custom_graph.get_tensor_by_name('AixOp_1:0')
            tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')

            # relu_op = custom_graph.get_tensor_by_name('res3d_relu/Relu:0')
            
            #test tranpose
            test_transpose_node = custom_graph.get_tensor_by_name("Transpose_to_NHWC:0")

            print ("Before importing graph def")

            tf.import_graph_def(tensor_def,
                                input_map={last_node_list[0]: tensor_transpose_NHWC},
                                name="")

        # with tf.Graph().as_default() as test_custom_graph:
        #     aix_op = custom_graph.get_tensor_by_name('AixOp_1:0')
        #     tensor_transpose_NHWC = tf.transpose(aix_op, [0, 2, 3, 1], name='Transpose_to_NHWC')

        #     # tf.import_graph_def(custom_graph.as_graph_def(), input_map={last_node_list[0]: tensor_transpose_NHWC}, name="")
        #     tf.import_graph_def(custom_graph.as_graph_def(), name="")


        #     tf.import_graph_def(tensor_def, input_map={last_node_list[0]: tensor_transpose_NHWC}, name="")


        # Merge graph all together
        # last_node_list = ["res3d_relu/Relu"]
        # output_node_list = [i.name for i in self.__output_tensors]

        # tensor_def = self.__axfc_util.extract_sub_graph(last_node_list, output_node_list)
        # with custom_graph.as_default() as custom_graph:
        #     # last_node_op = custom_graph.get_tensor_by_name("res4f_relu/Relu:0")
        #     # res3d_relu/Relu
        #     # last_node_op = custom_graph.get_tensor_by_name("input_1:0")
            
        #     # last_op = tensor_def.get_tensor_by_name("res5a_branch2a/convolution:0")
        #     # tf.import_graph_def(tensor_def,
        #     #                     input_map={"clipped_boxes/Shape:0":last_node_op}, 
        #     #                     name="")

        #     tf.import_graph_def(tensor_def, name="")
        
        custom_graph = self.__optimize_graph(custom_graph)  

        return custom_graph

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


    def __optimize_graph(self, custom_graph, aix_count=0):

        # take only tensor that has connection
        custom_graph_def = custom_graph.as_graph_def()

        # print (custom_graph_def.node[0], "index 0")

        # for index, node in enumerate(custom_graph_def.node):
        #     if "AixOp" == node.name:
        #         custom_graph_def.node[index].name = "AixOp_%s" % aix_count

        #     if "Transpose_to_NHWC" == node.name:
        #         custom_graph_def.node[index].input[0] = "AixOp_%s" %aix_count 
        # take only tensor that has connection
        # for index, op in enumerate(custom_graph.get_operations()):
            # if not op.inputs and not op.outputs[0].consumers():
            #     del custom_graph_def.node[index]
            #     break
            
            # if op.name == "AixOp_2":
            #     del custom_graph_def.node[index]

        #         # Testing
        #         print (op.name, "op del")
        
        #directly override the node with predecessor to first node
        for index, node in enumerate(custom_graph_def.node):
            if "input_1_1" in node.input:
                custom_graph_def.node[index].input[0] = "input_1"
            
            if "res3d_relu/Relu_1" in node.input:
                custom_graph_def.node[index].input[0] = "res3d_relu/Relu"
            
            if node.name == "P4_upsampled/Shape" or node.name == "C3_reduced/convolution":
                custom_graph_def.node[index].input[0] = "res3d_relu/Relu"
            

        #Test Directly Remove Unwated Nodes
        # unwated_node_list = ["input_1_1", "res4f/add", "res4a_branch2a/kernel", "padding_conv1/Pad_1", "Transpose_to_NCHW_2", "AixOp_2", "Transpose_to_NHWC_2", "res3d_relu/Relu_1"]    

        # Test statically remove invalid placeholder
        # for index, node in enumerate(custom_graph_def.node):

        #     if node.name == "res3d/add" or node.name == "input_1_2" or node.name == "input_1_1":
        #         del custom_graph_def.node[index]
                # break

        #     if node.name == "resnet_model/add_15":
        #         del custom_graph_def.node[index]
        #         break
            
            # if node.name in unwated_node_list:
            #     del custom_graph_def.node[index]
        
        
        # for index, op in enumerate(custom_graph.get_operations()):
        #     if op.name in unwated_node_list:
        #         del custom_graph_def.node[index]

        # for index, op in enumerat e(custom_graph.get_operations()):
        #     if not op.inputs and not op.outputs[0].consumers() and op.name != "Transpose_to_NCHW_1/perm":
        #         del custom_graph_def.node[index]
        
        with tf.Graph().as_default() as main_graph:
            tf.import_graph_def(custom_graph_def, name='')

        return main_graph