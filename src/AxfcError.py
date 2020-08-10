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

#######################################################################
# AxfcError enum class
#######################################################################

class AxfcError(enum.Enum):
    SUCCESS = 0
    INVALID_PARAMETER = 1
    INVALID_FILE_PATH = 2
    INVALID_INPUT_TYPE = 3
    INVALID_TF_GRAPH = 4
    INVALID_IR_GRAPH = 5
    EMPTY_IR_BLOCK = 6
    PRED_NODE_NOT_FOUND = 7
    INVALID_MD_FORMAT = 8
    NOT_AIXH_SUPPORT = 9
    NOT_IMPLEMENTED = 10
    UNKNOWN_TENSOR_TYPE = 11
    UNSUPPORTED_AIX_LAYER_EMIT = 12
    INVALID_AIX_LAYER_TYPE = 13
    INVALID_AIX_TENSOR_FORMAT = 14
    INVALID_CONVOLUTION_LAYER = 15
    INVALID_BATCHNORM_LAYER = 16
    INVALID_ACTIVATION_LAYER = 17
    INVALID_AIX_TENSOR_INPUT = 18