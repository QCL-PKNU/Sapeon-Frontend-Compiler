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

import enum
import json
import logging

from AxfcError import *

#######################################################################
# AIXLayer enum class (AIXGraph)
#######################################################################

class AIXLayerType(enum.Enum):
    AIX_LAYER_CONVOLUTION = 0
    AIX_LAYER_CONNECTED = 1
    AIX_LAYER_MAXPOOL = 2
    AIX_LAYER_AVGPOOL = 3
    AIX_LAYER_SOFTMAX = 4
    AIX_LAYER_ROUTE = 6
    AIX_LAYER_REORG = 7
    AIX_LAYER_EWADD = 8
    AIX_LAYER_UPSAMPLE = 9
    AIX_LAYER_PIXELSHUFFLE = 10
    AIX_LAYER_GROUP_CONV = 11
    AIX_LAYER_SKIP_CONV = 12
    AIX_LAYER_ACTIVATION = 13
    AIX_LAYER_BATCHNORM = 14
    AIX_LAYER_BIASADD = 15
    AIX_LAYER_OUTPUT = 16
    AIX_LAYER_INPUT = 17
    AIX_LAYER_WILDCARD = 18

#######################################################################
# AIXActivation enum class (AIXGraph)
#######################################################################

class AIXActivationType(enum.Enum):
    AIX_ACTIVATION_SIGMOID = 0
    AIX_ACTIVATION_RELU = 1
    AIX_ACTIVATION_LEAKY_RELU = 2
    AIX_ACTIVATION_PRELU = 3
    AIX_ACTIVATION_TANH = 4
    AIX_ACTIVATION_IDENTITY = 5

#######################################################################
# AIXSampling enum class (AIXGraph)
#######################################################################

class AIXSamplingType(enum.Enum):
    AIX_POOLING_MAX = 0
    AIX_POOLING_AVERAGE = 1
    AIX_POOLING_REORG = 2
    AIX_POOLING_UPSAMPLE = 3
    AIX_POOLING_PIXELSHUFFLE = 4

#######################################################################
# AxfcMachineDesc class
#######################################################################
class AxfcMachineDesc:

    # for input type
    TYPE_TENSORFLOW = 0
    TYPE_PYTORCH = 1
    TYPE_MXNET = 2
    TYPE_UNKNOWN = 3

    # for hardware acceleration
    DEFAULT_PROFIT_THRESHOLD = 1000

    ## @var __info
    # general info. of dictionary type for AIX compiler

    ## @var __aix_op
    # general machine description info. of dictionary type

    ## The constructor
    def __init__(self):
        self.__info = dict()
        self.__aix_ops = dict()

    ## This method is used to read a machine description from the given input path.
    # @param self this object
    # @param path file path of the machine description
    # @return error info
    def read_file(self, path: str) -> AxfcError:
        logging.info("AxfcMachineDesc:read_file - path: %s", path)

        # read machine description
        try:
            with open(path, "rt") as md_file:
                self.__info = json.load(md_file)
        except FileNotFoundError as e:
            logging.warning(str(e))
            return AxfcError.INVALID_FILE_PATH
        except ValueError as e:
            logging.warning(str(e))
            return AxfcError.INVALID_MD_FORMAT

        # make a dictionary of AIX operations
        try:
            self.__aix_ops = dict()

            for (op, info) in self.__info["AIX_OPERATION"].items():

                record = AxfcMachineDesc.AIXOpInfo(op)
                record.layer = info["layer"]
                record.activation = info["activation"]
                record.is_group = bool(info["is_group"])
                record.is_conv = bool(info["is_conv"])
                record.profit = int(info["profit"])

                self.__aix_ops[op] = record

        except ValueError as e:
            logging.warning(str(e))
            return AxfcError.INVALID_MD_FORMAT

        return AxfcError.SUCCESS

    ## This method indicates whether the given operation is supported by the AIX hardware or not.
    # @param self this object
    # @param op_name operation name
    # @return the input type of the frontend compilation
    def get_axih_support(self, op_name: str) -> bool:
        #logging.info("AxfcMachineDesc:check_axih_support - %s", op_name)
        try:
            if self.__aix_ops[op_name] is not None:
                return True
        except KeyError as e:
            logging.warning(e)
            return False

    ## This method returns the type of AI framework.
    # @param self this object
    # @return the input type of the frontend compilation
    def get_in_type(self):
        #logging.info("AxfcMachineDesc:get_in_type")
        try:
            in_type = self.__info["AIX_MODEL_TYPE"]

            if in_type == "TENSORFLOW":
                return AxfcMachineDesc.TYPE_TENSORFLOW
            else:
                return AxfcMachineDesc.TYPE_UNKNOWN
        except KeyError as e:
            logging.warning(e)
            return AxfcMachineDesc.TYPE_UNKNOWN

    ## This method returns the input type of the frontend compilation.
    # @param self this object
    # @return the profit threshold to determine whether to use hardware acceleration
    def get_profit_threshold(self):
        #logging.info("AxfcMachineDesc:get_profit_threshold")
        try:
            threshold = self.__info["AIX_PROFIT_THRESHOLD"]
            return int(threshold)
        except KeyError as e:
            logging.warning(e)
            return DEFAULT_PROFIT_THRESHOLD

    ## For debugging
    def __str__(self):
        str_buf = ">> Machine Description: \n" + str(self.__info)
        str_buf = ">> AIX Operations: \n" + str(self.__aix_ops)
        return str_buf + "\n"

    #######################################################################
    # AIXMDRecord inner class
    #######################################################################

    class AIXOpInfo:
        def __init__(self, op):
            self.op = op
            self.layer = None
            self.activation = None
            self.is_group = False
            self.is_conv = False
            self.profit = 0

        def __str__(self):
            str_buf = ">> AIX Operation Info.: " + str(self.op) + "\n"
            str_buf += "- layer: " + str(self.layer) + "\n"
            str_buf += "- activation: " + str(self.activation) + "\n"
            str_buf += "- group: " + str(self.is_group) + "\n"
            str_buf += "- is_conv: " + str(self.is_conv) + "\n"
            str_buf += "- profit: " + str(self.profit) + "\n"
            return str_buf
