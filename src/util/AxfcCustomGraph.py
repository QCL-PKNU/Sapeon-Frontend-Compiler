#######################################################################
#   AxfcCustomGraph
#
#   Created: 2020. 09. 10
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from util.AxfcTFGraphUtil import AxfcTFGraphUtil
from util.AxfcUtil import  *
import tensorflow as tf


#######################################################################
# AxfcCustomGraph class
#######################################################################

class AxfcCustomGraph:

    ## @var input_tensor_names 
    # the input layer tensor name

    ## @var last_subgraph_names 
    # the last unsupported subgraphs. 
    # It refer to the layers that not supported by AIX hardware/simulator. 

    ## @var __graph_def 
    # the object of graph_pb2.GraphDef which is loaded from tensorflow frozen model 

    ## @var __axfc_util 
    # the AxfcTFgraphUtil object

    ## @var __path_module 
    # the custom kernel library (*.so) path

    ## @var __output_type 
    # the type of output (normally tf.float32)

    ## @var __aix_graph_path 
    # the path of aix graph

    ## The constructor
    def __init__(self, input_tensor_names: list,
                 graph_def,
                 path_module: str,
                 output_type,
                 aix_graph_path: str,
                 last_subgraph_names = None
                 ):
        self.input_tensor_names = input_tensor_names
        self.last_subgraph_names = last_subgraph_names

        self.__graph_def = graph_def
        self.__axfc_util = AxfcTFGraphUtil(graph_def)
        self.__path_module = path_module
        self.__output_type = output_type
        self.__aix_graph_path = aix_graph_path

    ## This function is used to get the all inputs tensors 
    #
    # @return tuple(input_tensors, inputs_graph)
    #   - input_tensors the input tensors of the first layer
    #   - inputs_graph the subgraph that build by merging input tensors
    def __emit_input_tensors(self):
        input_tensors_graph = self.__axfc_util.extract_sub_graph_from_begin(self.__graph_def, self.input_tensor_names)

        with tf.Graph().as_default() as inputs_graph:
            tf.import_graph_def(input_tensors_graph, name='')

        input_tensors = []
        for tensor_name in self.input_tensor_names:
            tensor = inputs_graph.get_tensor_by_name("{}:0".format(tensor_name))
            input_tensors.append(tensor)

        return input_tensors, inputs_graph

    ## This method is used to load the aix custom kernel library object
    #
    # @return tuple(aix_tensor, aix_graph)
    #   - aix_tensor the custom aix tensor from custom kernel library
    #   - aix_graph the graph consisted aix_tensor
    def __load_aix_op(self):
        input_tensors, inputs_graph = self.__emit_input_tensors()

        tf.compat.v1.disable_eager_execution()

        op_module = tf.load_op_library(self.__path_module)

        with inputs_graph.as_default() as aix_graph:
            input_tensor = inputs_graph.get_operations()[-1].outputs[-1]
            tensor_transpose = tf.transpose(input_tensors[0], [0, 3, 1, 2])
            aix_tensor = op_module.aix_op(
                input=[tensor_transpose],
                output_type=self.__output_type,
                aix_graph_path=self.__aix_graph_path,
            )

        return aix_tensor, aix_graph

    ## This method is used to optimize subgraphs
    #
    # because there are unusable layers while manipulate the subgraphs.
    # @return main_graph the optimized graph
    def __optimize_graph(self, custom_graph):
        nodes = []
        # take only tensor that has connection
        custom_graph_def = custom_graph.as_graph_def()

        # take only tensor that has connection
        for index, op in enumerate(custom_graph.get_operations()):
            if not op.inputs and not op.outputs[0].consumers():
                del custom_graph_def.node[index]
                break

        with tf.Graph().as_default() as main_graph:
            tf.import_graph_def(custom_graph_def, name='')

        return main_graph

    ## This method is used to get the custom graph by rebuilting or merging all subgraphs
    #
    # @return custom_graph the final optimized graph
    def get_custom_graph(self):

        aix_tensor, aix_graph = self.__load_aix_op()

        if self.last_subgraph_names is not None:

            tensors_def = self.__axfc_util.extract_sub_graph(self.last_subgraph_names['input'],
                                                             self.last_subgraph_names['output'])


            with aix_graph.as_default() as custom_graph:

                # Add the new tensor to last in order to alter output data format from NCHW -> NHWC
                aix_op = aix_graph.get_tensor_by_name('AixOp:0')
                tf.transpose(aix_op, [0,2,3,1])

                tf.import_graph_def(tensors_def,
                                    input_map = {self.last_subgraph_names['input'][0]: aix_graph.get_tensor_by_name('transpose_1:0')},
                                    name='' )
                custom_graph.finalize()

            aix_graph = self.__optimize_graph(custom_graph)

        return aix_graph
