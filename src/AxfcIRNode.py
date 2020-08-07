#######################################################################
#   AxfcIRNode
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from AxfcError import *

#######################################################################
# AxfcIRNode class
#######################################################################
class AxfcIRNode:

    ## @var id
    # node ID

    ## @var op
    # operation of this node

    ## @var succs
    # node a list of successor nodes

    ## @var preds
    # node a list of predecessor nodes

    ## @var node_def
    # node a reference to an input node object

    ## @var is_aixh_support
    # indicate whether this node can be executed in hardware-manner

    ## @var is_root
    # indicate whether this node is root or not

    ## @var aixh_profit
    # specify the profit to be obtained by using AIXH

    ## @var block_ref
    # reference to the IR block that contains this node

    ## @var eval_flag
    # indicate whether this node has been already evaluated or not for maximal munching

    ## The constructor
    def __init__(self):
        self.__init__(None)

    def __init__(self, node_def):
        self.id = 0
        self.succs = list()
        self.preds = list()
        self.node_def = node_def
        self.block_ref = None

        self.aixh_profit = 0
        self.is_aixh_support = False
        self.is_root = False
        self.eval_flag = False

    ## This method is used to calculate and return the profit that we can get
    #  by accelerating the operation of this node in hardware-manner.
    #
    # @param self this object
    # @return the calculated profit
    def analyze_profit(self) -> int:

        if self.node_def is None:
            return 0

        # CHKME - YOUNGSUN (2020.08.06)
        # We must find the way to calculate the profit of a node for determining the hardware acceleration.

        return 500

    ## For debugging
    def __str__(self):
        str_buf = ">> IR Node: " + str(self.id) + ", " + self.op + "\n"
        str_buf += ">> Name: " + node_def.name + "\n"
        str_buf += ">> Pred: ["
        for pred in self.preds:
            str_buf += str(pred.id) + ", "
        str_buf += "]\n"

        str_buf += ">> Succ: ["
        for succ in self.succs:
            str_buf += str(succ.id) + ", "
        str_buf += "]\n"

        str_buf += ">> Attributes [root: " + str(self.is_root)
        str_buf += ", aixh_profit: " + str(self.aixh_profit)
        str_buf += ", aixh_support: " + str(self.is_aixh_support) + "]\n"

        return str_buf
