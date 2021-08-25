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
                 md:AxfcMachineDesc):
        self.__ir_graph = ir_graph
        self.__frozen_model_path = frozen_model_path
        self.__aix_graph_path = aix_graph_path
        self.__kernel_op_path = kernel_op_path
        self.__md = md

    # This method is used to build the custom graph
    # @return custom_graph the custom graph from the custom kernel library
    def get_custom_graph_v2(self):
        
        ir_blocks = [block for block in self.__ir_graph.blocks if block.is_aixh_support]

        graph_def = loadFrozenModel(self.__frozen_model_path)

        #remove training node
        graph_def = tf.compat.v1.graph_util.remove_training_nodes(graph_def, protected_nodes=None)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        
        # get input and output tensors
        input_tensors, output_tensors = analyze_inputs_outputs(graph)

        aix_custom_graph = AxfcCustomGraphV2(ir_blocks = ir_blocks,
                                            graph_def = graph_def,
                                            path_module = self.__kernel_op_path,
                                            output_type=tf.float32,
                                            aix_graph_path=self.__aix_graph_path,
                                            input_tensors = input_tensors,
                                            output_tensors = output_tensors,
                                            md = self.__md)
        
        return aix_custom_graph.get_custom_graph()
        
    ## This method is used to emit a launcher for the generated AIXGraph.
    # @param self this object
    def emit_aixh_launcher(self):
        logging.info("AxfcLauncherWriter:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass
