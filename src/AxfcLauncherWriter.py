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
        input_tensors, output_tensors = self.analyze_inputs_outputs(graph)

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
            'output':[output_tensors[0].name] # get the name of last tensor of AIXGraph
        }

        # get name of tensors
        input_names = [tensor.name for tensor in input_tensors]

        aix_custom_graph = AxfcCustomGraph(input_tensor_names=input_names,
                                           graph_def=graph_def,
                                           path_module=self.__kernel_op_path,
                                           last_subgraph_names=last_inout_tensors,
                                           output_type=tf.float32,
                                           aix_graph_path=self.__aix_graph_path,
                                           )
        return aix_custom_graph.get_custom_graph()

    # This method is used to analyze the graph to get inputs and outputs tensor
    # @param self this object
    # @param graph the tensor graph
    # @return inputs and outputs tensor operation
    def analyze_inputs_outputs(self, graph):
        ops = graph.get_operations()
        outputs_set = set(ops)
        inputs = []
        for op in ops:
            if len(op.inputs) == 0 and op.type != 'Const':
                inputs.append(op)
            else:
                for input_tensor in op.inputs:
                    if input_tensor.op in outputs_set:
                        outputs_set.remove(input_tensor.op)
        outputs = list(outputs_set)
        return (inputs, outputs)

    # This method is used to evaluate the custom graph
    # @param feed_input the input data (normally it's for import/input layer)
    # @param input_tensor_name the first input tensor name (default: take the first layer name)
    # @param last_tensor_name the last output tensor name (default: take the last layer name)
    # @return result_final the output value as numpy object
    def evaluate(self, feed_input):

        custom_graph = self.get_custom_graph()

        # Get the inputs and outputs of the graph
        inputs_tensor, outputs_tensor = self.analyze_inputs_outputs(custom_graph)

        # get input and outputs name
        input_tensor_name = '{}:0'.format(inputs_tensor[0].name)
        output_tensor_name = '{}:0'.format(outputs_tensor[0].name)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=custom_graph, config=config) as sess:
            result_final = sess.run(custom_graph.get_tensor_by_name(output_tensor_name),
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
