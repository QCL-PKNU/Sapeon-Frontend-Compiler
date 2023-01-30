
import logging

from AxfcError          import AxfcError
from .AxfcIRBuilder     import AxfcIRBuilder
from AxfcIRNode         import AxfcIRNode

import util.AxfcUtil    as _util

#######################################################################
# AxfcPTBuilder class
#######################################################################
class AxfcPTBuilder(AxfcIRBuilder):

    #The constructure
    def __init(self, md):
        super().__init__(md)

        #to store ir readout data
        self._ir_symtab_cnt = dict()
    
    #This function is used to read out the PyTorch mode
    #You may use PyTorch generic library to perform it.
    def _read_model_graph(self, path: str):
        return NotImplementedError()
    

    #This function is used to map the readout ir to AIX IR node
    #as well as building the computation AIX IR graph
    def _build_naive_ir(self, path: str) -> AxfcError:
        return NotImplementedError()