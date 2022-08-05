#######################################################################
#   AxfcFrontendCompiler
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################
from multiprocessing import Process
import os
from pathlib import Path
from AxfcONNXWriter import AxfcONNXWriter


from AxfcTFIRBuilder import *
from AxfcTFIRTranslator import *

from AxfcONNXIRBuilder import AxfcONNXIRBuilder
from AxfcONNXIRTranslator import AxfcONNXIRTranslator

from AxfcLauncherWriter import *
import glob
from util.AxfcUtil import *
from tensorflow.keras.preprocessing import image
from AxfcLauncher import *

#######################################################################
# AxfcFrontendCompiler
#######################################################################

class AxfcFrontendCompiler:
    NUM_CALIBRATION_DATA_ITEMS = 5

    ## @var __md
    # machine description object

    ## @var __calib_data
    # calibration data

    ## @var __ir_builder
    # AIXIR builder

    ## @var __ir_translator
    # AIXIR-to-AIXGraph translator

    ## The constructor
    def __init__(self):
        self.__md = None
        self.__calib_data = None
        self.__ir_builder = None
        self.__ir_translator = None

    ## This method returns the IR graph.
    #
    # @param self this object
    # @return the IR graph
    def get_ir_graph(self):
        return self.__ir_builder._ir_graph

    ## This method is used to read a machine description in the given path.
    #
    # @param self this object
    # @param path file path of AIXH machine description
    # @return error info
    def read_md_file(self, path: str) -> AxfcError:
        logging.info("AxfcFrontendCompiler:read_md_file - path: %s", path)

        self.__md = AxfcMachineDesc()
        return self.__md.read_file(path)

    ## This method is used to read calibration data in the given path.
    #
    # @param self this object
    # @param path file path to the calibration data
    # @return error info
    def read_calib_file(self, path: str) -> AxfcError:
        logging.info("AxfcFrontendCompiler:read_calib_file - path: %s", path)

        # if an external calibration data is not available, return without error
        if path is None:
            self.__calib_data = None
            return AxfcError.SUCCESS

        if not os.path.isfile(path):
            print("Invalid path to calibration data: " + path)
            return AxfcError.INVALID_CALIB_FILE_PATH

        # read a calibration data file and comprise the map of calibration data
        self.__calib_data = dict()

        fd = open(path, 'r')

        for calib_line in fd.readlines():
            # the format of calibration data
            # (0) index
            # (1) input name
            # (2) output name
            # (3) input calibration
            # (4) output calibration
            calib_data = calib_line.split()
            if len(calib_data) != self.NUM_CALIBRATION_DATA_ITEMS:
                return AxfcError.INVALID_CALIB_DATA_FORMAT

            input_name = calib_data[1]
            # output_name = calib_data[2]
            input_calib = float(calib_data[3])
            output_calib = float(calib_data[4])

            self.__calib_data[input_name] = {
                "input": input_calib,
                "output": output_calib
            }

        fd.close()

        return AxfcError.SUCCESS

    ## This method is used to compile an input AI network model into an AIXGraph object.
    #
    # @param self this object
    # @param path file path of an input AI network model
    # @return error info and an AXIGraph objects
    def compile(self, path: str) -> AxfcError:
        logging.info("AxfcFrontendCompiler:compile - path: %s", path)

        # create an IR builder of the given input type
        model_type = self.__md.get_model_type()

        if model_type is AxfcMachineDesc.TYPE_TENSORFLOW:
            self.__ir_builder = AxfcTFIRBuilder(self.__md)
            self.__ir_translator = AxfcTFIRTranslator(self.__md, path)
        
        elif model_type is AxfcMachineDesc.TYPE_ONNX:
            self.__ir_builder = AxfcONNXIRBuilder(self.__md)
            self.__ir_translator = AxfcONNXIRTranslator(self.__md, path)
            
        else:
            logging.warning("Not supported input type: %d", model_type)
            return AxfcError.INVALID_INPUT_TYPE, None

        # build AIXIR with the input graph
        err, aix_ir = self.__ir_builder.build_ir(path)
        if err is not AxfcError.SUCCESS:
            logging.warning("IR build error: %s", err)
            return err, None

        # perform the translation from AIXIR to AIXGraph
        err, aix_graphs = self.__ir_translator.emit_aixh_graphs(aix_ir, self.__calib_data)

        if err is not AxfcError.SUCCESS:
            logging.warning("IR-to-AIXGraph translation error: %s", err)
            return err, None

        return AxfcError.SUCCESS, aix_graphs

    ## This method is used to dump out the generated AIXGraphs.
    #
    # @param self this object
    # @param out_path a file path to output the AIXGraphs
    # @param aix_graphs a list of AIXGraphs to be dumped out
    # @return error info
    def dump_aix_graphs(self, out_path: str, aix_graphs: list, aix_graph_format:str) -> AxfcError:
        logging.info("AxfcIRTranslator:dump_aix_graphs - %s", out_path)
        if aix_graphs is None:
            logging.warning("No AIXGraphs found")
            return AxfcError.INVALID_AIX_GRAPH
        
        jobs = []
        for i, aix_graph in enumerate(aix_graphs):
            tmp_path = out_path + ".%s" % i

            p = Process(target=self.write_aix_graph, args=(tmp_path, aix_graph, aix_graph_format,))
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()

        return AxfcError.SUCCESS

    #For writing aix graph, implementation using multiprocess
    #data_mode can be BINARY or TEXT to write to output file. 
    def write_aix_graph(self, out_path: str, aix_graph: AIXGraph, data_mode:str) -> AxfcError:
        
        if not data_mode:
            data_mode == "BINARY"

        if data_mode.upper() == "BINARY":
            f = open(out_path, "wb")
            f.write(aix_graph.SerializeToString())
            f.close()
        elif data_mode.upper() == "TEXT":
            f = open(out_path, "wt")
            f.write(str(aix_graph))
            f.close()
        else:
            return AxfcError.INVALID_PARAMETER

        return  AxfcError.SUCCESS

# aix_graph.input_layers.append(aix_graph.layer[0].id)
    ## This method is used to dump out result of launcher
    #
    # @param self this object
    # @param path a file path of AI model
    # @param kernel_op_path a file path of custom operaiton kernel
    # @param aix_graph_path a file path of aix_graph
    # @param image_path a file path of an image for testing
    # @return result the value by evaluation model
    def dump_launcher(self, custom_model_path: str, kernel_op_path: str) -> AxfcError:

        # load multi image
        images = []

        path = '../tst/img/*'

        for f in glob.iglob(path):
            img = image.load_img(f, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)

        images = np.array(images)

        # for mobilenet
        # img_array_expanded_dims = tf.keras.applications.mobilenet.preprocess_input(np.expand_dims(img_array, axis=0))

        # for resnet model
        img_array_expanded_dims = tf.keras.applications.resnet50.preprocess_input(images)

        # evaluate custom model
        launcher = AxfcLauncher(custom_model_path, kernel_op_path)
        result = launcher.evaluate(feed_input=img_array_expanded_dims)

        # Evaluation and Prediction the model
        with open('../tst/ImageNetLabels.txt') as f:
            labels = [l.rstrip() for l in f]

        np_result = np.array(result)

        str_result = ''

        for res in np_result:
            sort_result = res.argsort()[-3:][::-1]
            for i in sort_result:
                str_result += '{0:0.3f}%'.format(res[i] * 100) + ' : ' +  str(labels[i]) + '\n'
            str_result += '\n'

        return str_result

    ## This method is used to dump out result of custom model
    #
    # @param self this object
    # @param path a file path of AI model
    # @param kernel_path a file path of custom operation kernel
    # @param aix_graph_path a file path of aix_graph
    # @param save_path a file path of storing file
    # @param file_name a file name of model
    # @return AxfcError
    def dump_custom_model(self, path: str, kernel_path: str, aix_graph_path: str, save_path: str):


        model_type = self.__md.get_model_type()

        if model_type is AxfcMachineDesc.TYPE_TENSORFLOW:
            #Leanghok: TensorFlow not supported for custom op and custom model at the moment
            logging.warning("TensorFlow is currently not supported for generating custom model. Only AIXGraph is generated.")
            return AxfcError.SUCCESS, None
            
            # AIX Launcher
            aix_launcher = AxfcLauncherWriter(frozen_model_path=path,
                                            aix_graph_path=aix_graph_path,
                                            kernel_op_path=kernel_path,
                                            ir_graph=self.get_ir_graph(),
                                            md = self.__md)

            # get custom graph model
            custom_graph_model = aix_launcher.get_custom_graph_v2()

            # write to file
            file_name = "custom_model"
            path = write2pb(custom_graph_model, des_path=save_path, name_file=file_name)

            if not Path(path).is_file():
                return AxfcError.INVALID_AIX_GRAPH, path

            return AxfcError.SUCCESS, path
        
        elif model_type is AxfcMachineDesc.TYPE_ONNX:
            
            onnx_writer = AxfcONNXWriter(frozen_model_path=path,
                                            aix_graph_path=aix_graph_path,
                                            kernel_op_path=kernel_path,
                                            ir_graph=self.get_ir_graph(),
                                            md = self.__md)
            
            err, out_path = onnx_writer.get_custom_graph()
            logging.info("AxfcFrontendCompiler:dump_custom_model - %s", out_path)

            return err, path
        
        else:
            # currently, we support only Tensorflow as an input type for the compilation
            logging.warning("Not supported input type: %d", model_type)
            return AxfcError.INVALID_INPUT_TYPE, None


    ## For debugging
    def __str__(self):
        pass
