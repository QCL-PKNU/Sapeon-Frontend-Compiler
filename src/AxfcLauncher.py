#######################################################################
#   AxfcLauncherWriter
#
#   Created: 2021. 02. 26
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from util.AxfcUtil import analyze_inputs_outputs, loadFrozenModel
import tensorflow as tf

#######################################################################
# AxfcLauncher class
#######################################################################


class AxfcLauncher:

    # @var __kernel_op_path
    # the path of op kernel

    # @var custom_model
    # the AIX freeze custom model

    ## The constructor
    # def __init__(self, custom_model_path: str, kernel_op_path: str):
    #     self.__kernel_op_path = kernel_op_path
    #     self.custom_model is TF model type

    ## The constructor
    def __init__(self, custom_model_path:str, kernel_op_path:str):

        custom_model_graph = loadFrozenModel(custom_model_path)

        tf.load_op_library(kernel_op_path)

        with tf.Graph().as_default() as custom_model:
            tf.import_graph_def(custom_model_graph, name='')

        self.custom_model = custom_model

    # This method is used to evaluate the custom graph
    # @param feed_input the input data (normally it's for import/input layer)
    # @param input_tensor_name the first input tensor name (default: take the first layer name)
    # @param last_tensor_name the last output tensor name (default: take the last layer name)
    # @return result_final the output value as numpy object
    def evaluate(self, feed_input):
        tf.keras.backend.clear_session()

        # Get the inputs and outputs of the graph
        inputs_tensor, outputs_tensor = analyze_inputs_outputs(self.custom_model)

        # get input and outputs name
        input_tensor_name = '{}:0'.format(inputs_tensor[0].name)
        output_tensor_name = '{}:0'.format(outputs_tensor[0].name)

        config = tf.compat.v1.ConfigProto()

        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.75

        with tf.compat.v1.Session(graph=self.custom_model, config=config) as sess:
            result_final = sess.run(self.custom_model.get_tensor_by_name(output_tensor_name),
                                    feed_dict={input_tensor_name: feed_input})

        return result_final
