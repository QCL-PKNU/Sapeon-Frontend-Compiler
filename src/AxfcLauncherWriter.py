#######################################################################
#   AxfcLauncherWriter
#
#   Created: 2020. 08. 15
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from AxfcIRGraph import *
from AxfcMachineDesc import *
from util import *


#######################################################################
# AxfcLauncherWriter class
#######################################################################

class AxfcLauncherWriter:

    # @var __frozen_model_path
    # the path of aix graph

    # @var __aix_graph_path
    # the path of aix graph

    # @var __kernel_op_path
    # the custom kernel library (*.so) path

    # @var __first_layer_tensor_name
    # the input layer tensor name

    # @var __last_inout_tensors
    # the last unsupported subgraphs.
    # it refers to the layers that not supported by AIX hardware/simulator.

    ## @var __md
    # AIX machine description

    ## @var __ir_graph
    # an AIXIR graph that will be used for writing the launcher

    ## The constructor
    # def __init__(self, md: AxfcMachineDesc, ir_graph: AxfcIRGraph):
    #     self.__md = md
    #     self.__ir_graph = ir_graph

    ## The constructor
    def __init__(self, frozen_model_path: str,
                 aix_graph_path: str,
                 kernel_op_path: str,
                 ir_graph: AxfcIRGraph,
                 ):
        self.__ir_graph = ir_graph
        self.__frozen_model_path = frozen_model_path
        self.__aix_graph_path = aix_graph_path
        self.__kernel_op_path = kernel_op_path
        self.__first_layer_tensor_name = 'import/{}'.format(ir_graph.blocks[0].nodes[0].name)
        # TODO: setup the last subgraph using ir_graph
        self.__last_inout_tensors = {'input': ['import/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd'],
                                     'output': ['import/MobilenetV1/Predictions/Reshape_1']}

    # This method is used to build the custom graph
    # @return custom_graph the custom graph from the custom kernel library
    def get_custom_graph(self):
        graph_def = loadFrozenModel(self.__frozen_model_path)
        axfc_util = AxfcTFGraphUtil(graph_def)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        first_layer_tensor = graph.get_tensor_by_name('{}:0'.format(self.__first_layer_tensor_name))
        input_names = [axfc_util.node_name(op.name) for op in first_layer_tensor.op.inputs]

        # last_inout_tensors =  {'input': ['import/MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd'],
        #                        'output': ['import/MobilenetV1/Predictions/Reshape_1']}

        aix_custom_graph = AxfcCustomGraph(input_tensor_names=input_names,
                                           graph_def=graph_def,
                                           path_module=self.__kernel_op_path,
                                           last_subgraph_names=self.__last_inout_tensors,
                                           output_type=tf.float32,
                                           aix_graph_path=self.__aix_graph_path,
                                           )
        return aix_custom_graph.get_custom_graph()

    # This method is used to evaluate the custom graph
    # @param feed_input the input data (normally it's for import/input layer)
    # @param input_tensor_name the first input tensor name (default: take the first layer name)
    # @param last_tensor_name the last output tensor name (default: take the last layer name)
    # @return result_final the output value as numpy object
    def evaluate(self, feed_input, input_tensor_name: str = None, last_tensor_name: str = None):
        custom_graph = self.get_custom_graph()
        input_tensor_name = 'import/input:0' if last_tensor_name is None else last_tensor_name
        last_tensor_name = '{}:0'.format(
            custom_graph.get_operations()[-1].name) if last_tensor_name is None else last_tensor_name

        with tf.compat.v1.Session(graph=custom_graph) as sess:
            result_final = sess.run(custom_graph.get_tensor_by_name(last_tensor_name),
                                    feed_dict={input_tensor_name: feed_input})
        return result_final

    ## This method is used to emit a launcher for the generated AIXGraph.
    # @param self this object
    def emit_aixh_launcher(self):
        logging.info("AxfcLauncherWriter:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass
