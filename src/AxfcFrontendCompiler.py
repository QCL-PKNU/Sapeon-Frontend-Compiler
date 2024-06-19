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
from CustomGraphs.AxfcONNXWriter import AxfcONNXWriter
from CustomGraphs.AxfcPTWriter import AxfcPTWriter

#TF Components
from Parsers.AxfcTFIRBuilder import *
from Translators.AxfcTFIRTranslator import *

#ONNX Components
from Parsers.AxfcONNXIRBuilder import AxfcONNXIRBuilder
from Translators.AxfcONNXIRTranslator import AxfcONNXIRTranslator

#PT Components
from Parsers.AxfcPTIRBuilder import AxfcPTIRBuilder
from Translators.AxfcPTIRTranslator import AxfcPTIRTranslator

from AxfcLauncherWriter import *
import glob
from util.AxfcUtil import *
from keras.preprocessing import image
from AxfcLauncher import *

#######################################################################
# AxfcFrontendCompiler
#######################################################################

class AxfcFrontendCompiler:
    """
    A compiler frontend for AIXH that handles reading machine descriptions, building IR graphs,
    and translating them into AIXGraph format.
    
    Attributes:
        NUM_CALIBRATION_DATA_ITEMS (int): Number of calibration data items.
        __md: Machine description object.
        __calib_data: Calibration data.
        __ir_builder: AIXIR builder object.
        __ir_translator: Object responsible for translating AIXIR to AIXGraph.
    """
    
    NUM_CALIBRATION_DATA_ITEMS = 5


    def __init__(self):
        self.__md = None
        self.__calib_data = None
        self.__ir_builder = None
        self.__ir_translator = None


    def get_ir_graph(self):
        """Return generated IRGraph"""
        return self.__ir_builder._ir_graph


    def read_md_file(self, path: str) -> AxfcError:
        """Reads a machine description from the specified file path.

        Args:
            path: The file path of the AIXH machine description to be read.
        """
        
        logging.info("AxfcFrontendCompiler:read_md_file - path: %s", path)

        self.__md = AxfcMachineDesc()
        return self.__md.read_file(path)


    def read_calib_file(self, path: str) -> AxfcError:
        """Reads calibration data.

        Args:
            path (str): A file path to the calibration data.
        """
        logging.info("AxfcFrontendCompiler:read_calib_file - path: %s", path)

        if not path:
            logging.warning("The path to calibration data is not specified.")
            self.__calib_data = None
            return AxfcError.SUCCESS

        if not os.path.isfile(path):
            print("Invalid path to calibration data: " + path)
            return AxfcError.INVALID_CALIB_FILE_PATH
        
        try:
            with open(path, 'r') as file:
                self.__calib_data = {}
                for calib_line in file.readlines():
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
        except Exception as e:
            return AxfcError.READ_FILE_ERROR
        
        logging.info("Calibration data read successfully from: %s", path)
        return AxfcError.SUCCESS
        

    def compile(self, path: str) -> AxfcError:
        """Compile an input AI model into AIXGraph object.

        Args:
            path (str): The file path of the input AI network model.

        Returns:
            A tuple containing an error code and an AIXGraph object. The AIXGraph object will
            be None if the compilation fails due to an unsupported input type or any error during
            the IR building or translation process.
        """
        logging.info("AxfcFrontendCompiler:compile - path: %s", path)

        model_type = self.__md.get_model_type()
        classname_map = {
            AxfcMachineDesc.TYPE_TENSORFLOW: (AxfcTFIRBuilder, AxfcTFIRTranslator),
            AxfcMachineDesc.TYPE_ONNX: (AxfcONNXIRBuilder, AxfcONNXIRTranslator),
            AxfcMachineDesc.TYPE_PYTORCH: (AxfcPTIRBuilder, AxfcPTIRTranslator)
        }
        builder_class, translator_class = classname_map.get(model_type, (None, None))
        if not builder_class or not translator_class:
            logging.warning("Unsupported input model type: %s", model_type)
            return AxfcError.INVALID_INPUT_TYPE, None
        
        self.__ir_builder = builder_class(self.__md)
        self.__ir_translator = translator_class(self.__md, path)

        # Build AIXIR from input graph of AI model
        err, aix_ir = self.__ir_builder.build_ir(path)
        if err is not AxfcError.SUCCESS:
            logging.warning("Failed to build IR: %s", err)
            return err, None
        
        # Translate AIXIR to AIXGraph
        err, aix_graphs = self.__ir_translator.emit_aixh_graphs(aix_ir, self.__calib_data)
        if err is not AxfcError.SUCCESS:
            logging.warning("Failed to translate IR to AIXGraph: %s", err)
            return err, None

        return AxfcError.SUCCESS, aix_graphs
    

    def dump_aix_graphs(self, out_path: str, aix_graphs: list, aix_graph_format: str) -> AxfcError:
        """Dump the generated AIXGraphs to the specified output path.

        Args:
            out_path: The file path to output the AIXGraphs.
            aix_graphs: A list of AIXGraphs to be dumped.
            aix_graph_format: The format to dump the AIXGraphs in.

        Returns:
            AxfcError: Error code indicating the success or failure of the operation.
        """
        for idx, aix_graph in enumerate(aix_graphs):
            tmp_path = f"{out_path}.{idx}"
            serialized_aix_graph = str(aix_graph)

            assert(isinstance(serialized_aix_graph, str))

            try:
                with open(tmp_path, "wt") as f:
                    f.write(serialized_aix_graph)
            except Exception as e:
                logging.error("Failed to write AIXGraph to file: %s", e)
                return AxfcError.WRITE_ERROR

        logging.info("AxfcFrontendCompiler: Successfully dumped all AIXGraphs.")
        return AxfcError.SUCCESS


    # def dump_aix_graphs(self, out_path: str, aix_graphs: list, aix_graph_format:str) -> AxfcError:
    #     """Dump the generated AIXGraphs to the specified output path.

    #     Args:
    #         out_path: The file path to output the AIXGraphs.
    #         aix_graphs: A list of AIXGraphs to be dumped.
    #         aix_graph_format: The format to dump the AIXGraphs in.

    #     Returns:
    #         AxfcError: Error code indicating the success or failure of the operation.
    #     """
    #     logging.info("AxfcIRTranslator:dump_aix_graphs - %s", out_path)
    #     if aix_graphs is None:
    #         logging.warning("No AIXGraphs found")
    #         return AxfcError.INVALID_AIX_GRAPH
        
    #     jobs = []
    #     for i, aix_graph in enumerate(aix_graphs):
    #         tmp_path = f"{out_path}.{i}.{aix_graph_format}"

    #         p = Process(target=self.write_aix_graph, args=(tmp_path, aix_graph, aix_graph_format,))
    #         jobs.append(p)
    #         p.start()

    #     for j in jobs:
    #         j.join()

    #     logging.info("Successfully dumped all AIXGraphs.")
    #     return AxfcError.SUCCESS


    # def write_aix_graph(self, out_path: str, aix_graph: AIXGraph, data_mode:str) -> AxfcError:
    #     """Writes the given AIXGraph to a file to a file in the specified data mode (either BINARY or TEXT)

    #     Args:
    #         out_path: The file path to output the AIXGraphs.
    #         aix_graphs: A list of AIXGraphs to be dumped.
    #         aix_graph_format: The format to dump the AIXGraphs in.
    #     """
        
    #     if not data_mode:
    #         data_mode = "BINARY"

    #     if data_mode.upper() == "BINARY":
    #         f = open(out_path, "wb")
    #         f.write(aix_graph.SerializeToString())
    #         f.close()
    #     elif data_mode.upper() == "TEXT":
    #         f = open(out_path, "wt")
    #         f.write(str(aix_graph))
    #         f.close()
    #     else:
    #         return AxfcError.INVALID_PARAMETER

    #     return  AxfcError.SUCCESS


    def load_images_from_folder(folder_path: str, target_size=(224, 224)) -> np.ndarray:
        """Loads images from a specified folder, converting them into an array of appropriate format for model evaluation.

        Args:
            folder_path: Path to the folder containing images.
            target_size: Target size to which images will be resized.

        Returns:
            A numpy array of the loaded and preprocessed images.
        """
        images = []
        for file_path in glob.iglob(folder_path):
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
        return np.array(images)


    def dump_launcher(self, custom_model_path: str, kernel_op_path: str) -> AxfcError:
        """Evaluates a custom model with images loaded from a specified path.
        
        Args:
            custom_model_path: Path to the custom model.
            kernel_op_path: Path to the custom operation kernel.
            images_path: Path to the images for testing.
            
        Returns:
            A string summarizing the top predictions for each image.
        """

        path = '../tst/img/*'
        images = self.load_images_from_folder(path)


        for f in glob.iglob(path):
            img = image.load_img(f, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)

        images = np.array(images)
        preprocessed_images = tf.keras.applications.resnet50.preprocess_input(images)

        # Evaluate custom model
        launcher = AxfcLauncher(custom_model_path, kernel_op_path)
        results = launcher.evaluate(feed_input=preprocessed_images)

        # Load label
        with open('../tst/ImageNetLabels.txt') as file:
            labels = [line.rstrip() for line in file]

        # Prepare result summary
        summary = ''
        for result in results:
            top_indices = result.argsort()[-3:][::-1]
            for index in top_indices:
                summary += f'{result[index] * 100:0.3f}% : {labels[index]}\n'
            summary += '\n'

        return summary


    def dump_custom_model(self, path: str, kernel_path: str, aix_graph_path: str, save_path: str):
        """Dumps the custom model based on the model type and provided paths.

        Args:
            path: The file path of the AI model.
            kernel_path: The file path of the custom operation kernel.
            aix_graph_path: The file path of the AIXGraph.
            save_path: The directory path where the model should be stored.

        Returns:
            A tuple of (AxfcError, Optional[str]): Error code and the path of the saved model or None.
        """

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
        
        elif model_type is AxfcMachineDesc.TYPE_PYTORCH:

            pt_writer = AxfcPTWriter(frozen_model_path=path,
                                     aix_graph_path=aix_graph_path,
                                    #  kernel_op_path=kernel_path,
                                     ir_graph=self.get_ir_graph(),
                                     md=self.__md)

            err, out_path = pt_writer.get_custom_graph()
            logging.info("AxfcFrontendCompiler:dump_custom_model - %s", out_path)

            return err, path
        
        else:
            # currently, we support only Tensorflow as an input type for the compilation
            logging.warning("Not supported input type: %d", model_type)
            return AxfcError.INVALID_INPUT_TYPE, None


    ## For debugging
    def __str__(self):
        pass
