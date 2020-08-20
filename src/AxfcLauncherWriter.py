#######################################################################
#   AxfcLauncherWriter
#
#   Created: 2020. 08. 15
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from AxfcIRGraph import *
from AxfcMachineDesc import *

#######################################################################
# AxfcLauncherWriter class
#######################################################################

class AxfcLauncherWriter:

    ## @var __md
    # AIX machine description

    ## @var __ir_graph
    # an AIXIR graph that will be used for writing the launcher

    ## The constructor
    def __init__(self, md: AxfcMachineDesc, ir_graph: AxfcIRGraph):
        self.__md = md
        self.__ir_graph = ir_graph

    ## This method is used to emit a launcher for the generated AIXGraph.
    # @param self this object
    def emit_aixh_launcher(self):
        logging.info("AxfcLauncherWriter:emit_aixh_launcher")
        pass

    ## For debugging
    def __str__(self):
        pass
