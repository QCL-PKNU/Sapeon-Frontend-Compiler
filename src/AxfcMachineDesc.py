#######################################################################
#   AxfcMachineDesc
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import json
import logging

from AxfcError import *


#######################################################################
# AxfcMachineDesc class
#######################################################################

class AxfcMachineDesc:
    # for input type
    TYPE_UNKNOWN = 0
    TYPE_TENSORFLOW = 1
    TYPE_PYTORCH = 2
    TYPE_MXNET = 3
    TYPE_ONNX = 4
    

    # for hardware acceleration
    DEFAULT_PROFIT_THRESHOLD = 1000

    # for compiling break point condition
    BREAK_POINT_CONDITION = False

    ## @var __aix_model_info_tbl
    # general info. of dictionary type for AIX compiler

    ## @var __aix_layer_info_tbl
    # general machine description info. of dictionary type

    ## The constructor
    def __init__(self):
        self.__aix_model_info_tbl = dict()
        self.__aix_layer_info_tbl = dict()

    ## This method is used to read a machine description from the given input path.
    #
    # @param self this object
    # @param path file path of the machine description
    # @return error info
    def read_file(self, path: str) -> AxfcError:
        logging.info("AxfcMachineDesc:read_file - path: %s", path)

        # read machine description
        try:
            with open(path, "rt") as md_file:
                self.__aix_model_info_tbl = json.load(md_file)
        except FileNotFoundError as e:
            logging.warning("read_file1: %s", str(e))
            return AxfcError.INVALID_FILE_PATH
        except ValueError as e:
            logging.warning("read_file2: %s", str(e))
            return AxfcError.INVALID_MD_FORMAT

        # make a dictionary of AIX operations
        try:
            self.__aix_layer_info_tbl = dict()

            for (layer_type, layer_info) in self.__aix_model_info_tbl["AIX_LAYER"].items():
                aix_layer_info = AxfcMachineDesc.AIXLayerInfo(layer_type)
                aix_layer_info.layer = layer_info["layer"]
                aix_layer_info.activation = layer_info["activation"]
                aix_layer_info.is_group = bool(layer_info["is_group"])
                aix_layer_info.is_conv = bool(layer_info["is_conv"])
                aix_layer_info.profit = int(layer_info["profit"])

                self.__aix_layer_info_tbl[layer_type] = aix_layer_info

        except ValueError as e:
            logging.warning("read_file3: %s", str(e))
            return AxfcError.INVALID_MD_FORMAT

        return AxfcError.SUCCESS

    ## This method returns the information of a specific AIX layer.
    #
    # @param self this object
    # @param layer_type the name of an AIX layer type to be returned
    # @return an operation information
    def get_layer_info(self, layer_type: str):
        # logging.info("AxfcMachineDesc:get_layer_info - %s", layer_type)
        try:
            return self.__aix_layer_info_tbl[layer_type]
        except KeyError:
            return None

    ## This method indicates whether the given operation is supported by the AIX hardware or not.
    #
    # @param self this object
    # @param layer_type the name of an AIX layer type to be checked
    # @return the input type of the frontend compilation
    def get_aixh_support(self, layer_type: str) -> bool:
        # logging.info("AxfcMachineDesc:get_aixh_support - %s", op)
        if self.get_layer_info(layer_type) is not None:
            return True
        else:
            return False

    ## This method returns the type of AI framework.
    #
    # @param self this object
    # @return the input type of the frontend compilation
    def get_model_type(self):
        # logging.info("AxfcMachineDesc:get_model_type")
        try:
            model_type = self.__aix_model_info_tbl["AIX_MODEL_TYPE"]

            if model_type.upper() == "TENSORFLOW":
                return AxfcMachineDesc.TYPE_TENSORFLOW
            elif model_type.upper() == "ONNX":
                return AxfcMachineDesc.TYPE_ONNX
            elif model_type.upper() == "PYTORCH":
                return AxfcMachineDesc.TYPE_PYTORCH
            else:
                return AxfcMachineDesc.TYPE_UNKNOWN
        except KeyError as e:
            logging.warning("get_model_type: %s", str(e))
            return AxfcMachineDesc.TYPE_UNKNOWN

    ## This method returns the name of the input AI model.
    #
    # @param self this object
    # @return the model name
    def get_model_name(self) -> str:
        # logging.info("AxfcMachineDesc:get_model_name")
        try:
            model_name = self.__aix_model_info_tbl["AIX_MODEL_NAME"]
        except KeyError as e:
            logging.warning("get_model_name: %s", str(e))
            return "NO_NAME"

        return model_name

    ## This method returns the input type of the frontend compilation.
    #
    # @param self this object
    # @return the profit threshold to determine whether to use hardware acceleration
    def get_profit_threshold(self):
        # logging.info("AxfcMachineDesc:get_profit_threshold")
        try:
            threshold = self.__aix_model_info_tbl["AIX_PROFIT_THRESHOLD"]
            AxfcMachineDesc.DEFAULT_PROFIT_THRESHOLD = int(threshold)
            return int(threshold)
        except KeyError as e:
            logging.warning("get_profit_threshold: %s", str(e))
            return AxfcMachineDesc.DEFAULT_PROFIT_THRESHOLD

    def get_break_point_node(self):
        try:
            break_point_node = self.__aix_model_info_tbl["STOP_COMPILING_POINT"]
            return break_point_node
        except KeyError as e:
            logging.warning("get_break_point_node: %s", str(e))
            return ""

    def get_input_point(self):
        try:
            input_point = self.__aix_model_info_tbl["INPUT_POINT"]
            return input_point
        except KeyError as e:
            logging.warning("get_input_point: %s", str(e))
            return ""

    ## For debugging
    def __str__(self):
        str_buf = ">> Machine Description: \n" + str(self.__aix_model_info_tbl)
        str_buf += ">> AIX Operations: \n" + str(self.__aix_layer_info_tbl)
        return str_buf + "\n"

    #######################################################################
    # AIXLayerInfo inner class
    #######################################################################


    class AIXLayerInfo:
        ## @var op
        # layer operation name of the layer info

        ## @var layer
        # AIX layer ID of the layer info

        ## @var activation
        # AIX activation ID of the layer info

        ## @var is_group
        # indicate whether this layer is group layer or not

        ## @var is_conv
        # indicate whether this layer is convolution layer or not

        ## @var profit
        # the profit that can be obtained by accelerating this layer using AIXH

        ## The constructor
        def __init__(self, op):
            self.op = op
            self.layer = None
            self.activation = None
            self.is_group = False
            self.is_conv = False
            self.profit = 0

        ## For debugging
        def __str__(self):
            str_buf = ">> AIX Operation Info.: " + str(self.op) + "\n"
            str_buf += "- layer: " + str(self.layer) + "\n"
            str_buf += "- activation: " + str(self.activation) + "\n"
            str_buf += "- is_group: " + str(self.is_group) + "\n"
            str_buf += "- is_conv: " + str(self.is_conv) + "\n"
            str_buf += "- profit: " + str(self.profit) + "\n"
            return str_buf
