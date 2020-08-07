#######################################################################
#   AxfcTFIRTranslator
#
#   Created: 2020. 08. 07
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import enum
import logging

from aixh_pb2 import *
from AxfcError import *
from AxfcIRNode import *
from AxfcIRBlock import *
from AxfcIRGraph import *
from AxfcMachineDesc import *
from AxfcIRTranslator import *

#######################################################################
# AxfcTFIRTranslator class
#######################################################################
class AxfcTFIRTranslator(AxfcIRTranslator):

    ## The constructor
    def __init__(self, md):
        super().__init__(md)

    def _emit_aixh_node(self, ir_node: AxfcIRNode) -> {AxfcError, AIXLayer}:
        return AxfcError.NOT_IMPLEMENTED_YET, None
