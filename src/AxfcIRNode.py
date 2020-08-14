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

    ## @var layer_id
    # node layer ID

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

    ## @var is_input
    # indicate whether this node is an input node or not

    ## @var is_output
    # indicate whether this node is an output node or not

    ## @var aixh_profit
    # specify the profit to be obtained by using AIXH

    ## @var block_ref
    # reference to the IR block that contains this node

    ## @var eval_flag
    # indicate whether this node has been already evaluated or not for maximal munching

    ## @var aix_layer
    # reference to the AIX layer derived from this node

    ## The constructor
    def __init__(self):
        self.__init__(None)

    def __init__(self, node_def):
        self.id = 0
        self.name = ""
        self.layer_id = 0
        self.succs = list()
        self.preds = list()
        self.node_def = node_def
        self.block_ref = None

        self.aixh_profit = 0
        self.is_aixh_support = False
        self.eval_flag = False

        self.is_input = False
        self.is_output = False

        self.aix_layer = None

    ## This method is used to calculate and return the profit that we can get
    #  by accelerating the operation of this node in hardware-manner.
    #
    # @param self this object
    # @return the calculated profit
    def analyze_profit(self) -> int:

        if self.node_def is None:
            return 0

        # CHKME - YOUNGSUN (2020.08.06)
        # We must find the way to calculate the profit of a node for determining
        # to use the hardware acceleration.

        return self.aixh_profit

    ## This methods is used to compare id with equal (==) for using Set
    # 
    # @param self this object
    # @param other another AxfcIRNode object
    def __eq__(self, other):
        
        # check other and self type 
        if not isinstance(other, type(self)): return NotImplemented

        return self.id == other.id

    ## This methods make this object become hasable by id
    # 
    # @param self this object
    def __hash__(self):
        return hash(self.id)

    ## For debugging
    def __str__(self):
        str_buf = ">> IR Node: " + str(self.id) + ", " + self.op + "\n"
        str_buf += ">> Name: " + self.name + "\n"
        str_buf += ">> Pred: ["
        for pred in self.preds:
            str_buf += str(pred.id) + ", "
        str_buf += "]\n"

        str_buf += ">> Succ: ["
        for succ in self.succs:
            str_buf += str(succ.id) + ", "
        str_buf += "]\n"

        str_buf += ">> Attributes ["
        str_buf += "is_input: " + str(self.is_input)
        str_buf += ", is_output: " + str(self.is_output)
        str_buf += ", aixh_profit: " + str(self.aixh_profit)
        str_buf += ", aixh_support: " + str(self.is_aixh_support) + "]\n"

        return str_buf
