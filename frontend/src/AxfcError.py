#######################################################################
#   AxfcError
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

from enum import auto


#######################################################################
# AxfcError enum class
#######################################################################

class AxfcError(enum.Enum):
    SUCCESS = 0
    INVALID_PARAMETER = 1
    INVALID_FILE_PATH = 2

    # Input model and graph
    INVALID_INPUT_TYPE = auto()
    INVALID_TF_GRAPH = auto()
    INVALID_ONNX_GRAPH = auto()
    INVALID_PT_GRAPH = auto()
    INVALID_IR_GRAPH = auto()
    EMPTY_IR_BLOCK = auto()
    PRED_NODE_NOT_FOUND = auto()

    # Machine description
    INVALID_MD_FORMAT = auto()
    NOT_AIXH_SUPPORT = auto()
    NOT_IMPLEMENTED = auto()
    UNKNOWN_TENSOR_TYPE = auto()

    # AIXGraph
    UNSUPPORTED_AIX_LAYER_EMIT = auto()
    INVALID_AIX_LAYER_TYPE = auto()
    INVALID_AIX_TENSOR_FORMAT = auto()
    INVALID_CONVOLUTION_LAYER = auto()
    INVALID_GROUP_CONV_LAYER = auto()
    INVALID_BATCHNORM_LAYER = auto()
    INVALID_ACTIVATION_LAYER = auto()
    INVALID_IDENTITY_LAYER = auto()
    INVALID_PAD_LAYER = auto()
    INVALID_AIX_TENSOR_INPUT = auto()
    DUMP_IR_GRAPH_ERROR = auto()
    INVALID_AIX_GRAPH = auto()
    INVALID_MAXPOOL_LAYER = auto()
    INVALID_EWADD_LAYER = auto()
    IVNALID_BIASADD_LAYER = auto()
    UNREMOVED_IDENTITY = auto()

    # Calibration data
    INVALID_CALIB_FILE_PATH = auto()
    INVALID_CALIB_DATA_FORMAT = auto()
    
    # Valdation
    INVALID_NODE = auto()

    # I/O
    READ_FILE_ERROR = auto()
    WRITE_ERROR = auto()