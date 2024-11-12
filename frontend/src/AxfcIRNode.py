#######################################################################
# AxfcIRNode class
#######################################################################

class AxfcIRNode:

    """
    Represents a node within an Axfc Intermediate Representation (IR) Graph.

    Attributes:
        id (int): ID of the node.
        name (str): Name of the node.
        layer_id (int): ID of the layer.
        op (str): Opteration of the node.
        succs (List[AxfcIRNode]): List of successor nodes.
        preds (List[AxfcIRNode]): List of predecessor nodes.
        node_def: Reference to the input node object definition.
        is_aixh_support (bool): Indicates if the node can be executed in hardware.
        is_input (bool): Indicates if the node is an input node.
        is_output (bool): Indicates if the node is an output node.
        aixh_profit (int): Profit obtained by using hardware acceleration (AIXH).
        block_ref: Reference to the IR block that contains this node.
        eval_flag (bool): Indicates if the node has been evaluated for maximal munching.
        aix_layer: Reference to the AIX layer derived from this node.
    """

    def __init__(self, node_def):
        self.id = 0
        self.name = ""
        self.output_name = ""
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
        self.op        = None


    def analyze_profit(self) -> int:
        """Calculates and returns the profit of accelerating this node's operation in hardware.

        Returns:
            The calculated profit for hardware acceleration.
        """

        if self.node_def is None:
            return 0

        # CHKME - YOUNGSUN (2020.08.06)
        # We must find the way to calculate the profit of a node for determining
        # to use the hardware acceleration.

        return self.aixh_profit



    def __eq__(self, other):
        """Check equality based on the 'id' attribute of two AxfcIRNode instances.

        This mothod check if other has the same type of AIXIRNode as current object.

        Args:
            other (AIXIRNode): Another object of AIXIRNode.

        Returns:
            value (bool): True if the instances have the same 'id', False otherwise. 
        """

        if not isinstance(other, type(self)):
            return NotImplemented

        return self.id == other.id
    

    ## This methods make this object become hasable by id
    # 
    # @param self this object
    def __hash__(self):
        return hash(self.id)

    ## Destructor
    def __del__(self):
        self.id = -1
        self.op = None
        self.name = None
        self.preds.clear()
        self.succs.clear()
        self.node_def = None


    def __str__(self):
        """Returns a string representation of the AIXIRNode instance."""
        
        pred_ids = ', '.join(str(pred.id) for pred in self.preds)
        succ_ids = ', '.join(str(succ.id) for succ in self.succs)

        return (f">> IR Node: {self.id}, {self.op}\n"
                f">> Name: {self.name}\n"
                f">> Pred: [{pred_ids}]\n"
                f">> Succ: [{succ_ids}]\n"
                f">> Attributes ["
                f"is_input: {self.is_input}, "
                f"is_output: {self.is_output}, "
                f"aixh_profit: {self.aixh_profit}, "
                f"aixh_support: {self.is_aixh_support}]")