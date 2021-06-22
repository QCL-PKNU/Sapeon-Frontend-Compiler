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
                 ):
        self.__ir_graph = ir_graph
        self.__frozen_model_path = frozen_model_path
        self.__aix_graph_path = aix_graph_path
        self.__kernel_op_path = kernel_op_path


    # This method is used to build the custom graph
    # @return custom_graph the custom graph from the custom kernel library
    def get_custom_graph(self):

        # filter get block that run in AIX
        ir_blocks = [ block for block in self.__ir_graph.blocks if block.is_aixh_support]

        graph_def = loadFrozenModel(self.__frozen_model_path)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        # get input and output tensors
        input_tensors, output_tensors = analyze_inputs_outputs(graph)

        # FOR TESTING
        # aix_graph = AIXGraph()
        # with open(self.__aix_graph_path, 'rb') as file:
        #     text_format.Parse(file.read(), aix_graph)
        #
        # input_name = aix_graph.layer[-1].name

        if len(ir_blocks) < 1:
            logging.error('Error] There is no block supported, Change decrease AIX_PROFIT_THRESHOLD in md file.')
            return graph

        input_name = ir_blocks[0].nodes[-1].name

        last_inout_tensors = {
            'input':[input_name], # get the name of last tensor that supported by AIX
            'output':[i.name for i in output_tensors] # get the name of last tensors of AIXGraph
        }

        # get name of tensors
        # input_names = [tensor.name for tensor in input_tensors]

        #get the name of the input tensors connecting to AixOp graph
        input_names = ir_blocks[0].nodes[0].node_def.input

        aix_custom_graph = AxfcCustomGraph(input_tensor_names=input_names,
                                           graph_def=graph_def,
                                           path_module=self.__kernel_op_path,
                                           last_subgraph_names=last_inout_tensors,
                                           output_type=tf.float32,
                                           aix_graph_path=self.__aix_graph_path,
                                           )
        return aix_custom_graph.get_custom_graph()

    def get_custom_graph_v2(self):
        
        ir_blocks = [block for block in self.__ir_graph.blocks if block.is_aixh_support]

        graph_def = loadFrozenModel(self.__frozen_model_path)

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
                                            output_tensors = output_tensors)
        
        return aix_custom_graph.get_custom_graph()
        
    ## This method is used to emit a launcher for the generated AIXGraph.
    # @param self this object
    def emit_aixh_launcher(self):
        logging.info("AxfcLauncherWriter:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass
