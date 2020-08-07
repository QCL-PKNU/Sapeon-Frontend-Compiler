#######################################################################
#   AxfcError
#
#   Created: 2020. 08. 03

#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Copyright (c) 2020
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from enum import Enum

##
# AxfcError
#
class AxfcError(Enum):
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