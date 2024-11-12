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
    TYPE_UNKNOWN = 0
    TYPE_TENSORFLOW = 1
    TYPE_PYTORCH = 2
    TYPE_MXNET = 3
    TYPE_ONNX = 4
    

    # for hardware acceleration
    DEFAULT_PROFIT_THRESHOLD = 1000

    # for compiling break point condition
    BREAK_POINT_CONDITION = False


    def __init__(self):
        """Instantiate object of AxfcMachineDesc.

        Attributes:
            __aix_model_info_tbl (dict): A dictionary contains configuration related to model.
            __aix_layer_info_tbl (dict): A dictionary contains information of supported layer.
        """
        self.__aix_model_info_tbl = dict()
        self.__aix_layer_info_tbl = dict()


    def read_file(self, path: str) -> AxfcError:
        """Read a machine description.
        Args:
            path (str): A file path of machine description.
        
        Returns:
            error (AxfcError): An error status indicates failure or success.
        """
        logging.info("AxfcMachineDesc:read_file - path: %s", path)

        try:
            with open(path, "rt") as md_file:
                self.__aix_model_info_tbl = json.load(md_file)
        except FileNotFoundError as e:
            logging.warning("read_file1: %s", str(e))
            return AxfcError.INVALID_FILE_PATH
        except ValueError as e:
            logging.warning("read_file2: %s", str(e))
            return AxfcError.INVALID_MD_FORMAT

        # Check for required keys in the loaded JSON
        required_keys = ["AIX_MODEL_TYPE", "AIX_LAYER"]
        for key in required_keys:
            if key not in self.__aix_model_info_tbl:
                logging.warning("read_file3: Missing required key: %s", key)
                return AxfcError.INVALID_MD_FORMAT

        # Process AIX Layers (operations) information
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

        except (ValueError, KeyError) as e:
            logging.warning("read_file4: %s", str(e))
            return AxfcError.INVALID_MD_FORMAT

        return AxfcError.SUCCESS
    

    def get_layer_info(self, layer_type: str):
        """Returns then information of a specific AIX Layer
        
        Args:
            layer_type (str): The name of an AIX layer type.
        """
        # logging.info("AxfcMachineDesc:get_layer_info - %s", layer_type)
        try:
            return self.__aix_layer_info_tbl[layer_type]
        except KeyError:
            return None


    def get_aixh_support(self, layer_type: str) -> bool:
        """Indicates whether the given operation is supported by the AIX hardware.

        Args:
            layer_type: The name of an AIX layer type to be checked.
        
        """
        # logging.info("AxfcMachineDesc:get_aixh_support - %s", op)
        return self.get_layer_info(layer_type) is not None


    def get_model_type(self):
        """Returns the type of AI framework being used.
        
        Returns:
            str: The AI framework type.
        """
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
        

    def get_model_name(self) -> str:
        """Return the name of the input AI model"""
        # logging.info("AxfcMachineDesc:get_model_name")
        try:
            model_name = self.__aix_model_info_tbl["AIX_MODEL_NAME"]
        except KeyError as e:
            logging.warning("get_model_name: %s", str(e))
            return "NO_NAME"

        return model_name


    def get_profit_threshold(self):
        """
        Returns the profit threshold to determine whether to use hardware acceleration.

        Returns:
            int: The profit threshold value.
        """
        logging.info("AxfcMachineDesc:get_profit_threshold")
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


    def __str__(self):
        """Returns a string representation of the AIXMachineDesc instance."""
        
        return (f">> Machine Description: \n{self.__aix_model_info_tbl}\n"
                f">> AIX Operations: \n{self.__aix_layer_info_tbl}\n")

    #######################################################################
    # AIXLayerInfo inner class
    #######################################################################


    class AIXLayerInfo:


        def __init__(self, op):
            """Instantiates an AIXLayerInfo object with provided layer details.

            Attributes:
                op (str): Operation name of the layer.
                layer (int or None): AIX layer ID.
                activation (int or None): AIX activation ID.
                is_group (bool): Indicates whether the layer is a group layer.
                is_conv (bool): Indicates whether the layer is a convolution layer.
                profit (float): The profit from accelerating this layer using AIX hardware.
            """
            self.op = op
            self.layer = None
            self.activation = None
            self.is_group = False
            self.is_conv = False
            self.profit = 0


        def __str__(self):
            """Returns a string representation of the AIXLayerInfo instance."""
            
            return (f">> AIX Operation Info.: {self.op}\n"
                    f"- layer: {self.layer}\n"
                    f"- activation: {self.activation}\n"
                    f"- is_group: {self.is_group}\n"
                    f"- is_conv: {self.is_conv}\n"
                    f"- profit: {self.profit}\n")
