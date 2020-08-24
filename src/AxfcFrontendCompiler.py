#######################################################################
#   AxfcFrontendCompiler
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

from AxfcTFIRBuilder import *
from AxfcTFIRTranslator import *
from AxfcLauncherWriter import *

#######################################################################
# AxfcFrontendCompiler
#######################################################################

class AxfcFrontendCompiler:

    ## @var __md
    # machine description object

    ## @var __ir_builder
    # AIXIR builder

    ## @var __ir_translator
    # AIXIR-to-AIXGraph translator

    ## The constructor
    def __init__(self):
        self.__md = None
        self.__ir_builder = None
        self.__ir_translator = None

    ## This method returns the IR graph.
    #
    # @param self this object
    # @return the IR graph
    def get_ir_graph(self):
        return self.__ir_builder._ir_graph

    ## This method is used to read a machine description in the given path.
    #
    # @param self this object
    # @param path file path of AIXH machine description
    # @return an AXIGraph object
    def read_md_file(self, path: str) -> AxfcError:
        logging.info("AxfcFrontendCompiler:read_md_file - path: %s", path)

        self.__md = AxfcMachineDesc()
        return self.__md.read_file(path)

    ## This method is used to compile an input AI network model into an AIXGraph object.
    #
    # @param self this object
    # @param path file path of an input AI network model
    # @return error info and an AXIGraph objects
    def compile(self, path: str) -> AxfcError:
        logging.info("AxfcFrontendCompiler:compile - path: %s", path)

        # create an IR builder of the given input type
        in_type = self.__md.get_in_type()

        if in_type is AxfcMachineDesc.TYPE_TENSORFLOW:
            self.__ir_builder = AxfcTFIRBuilder(self.__md)
            self.__ir_translator = AxfcTFIRTranslator(self.__md)
        else:
            # currently, we support only Tensorflow as an input type for the compilation
            logging.warning("Not supported input type: %d", in_type)
            return AxfcError.INVALID_INPUT_TYPE, None

        # build AIXIR with the input graph
        err, aix_ir = self.__ir_builder.build_ir(path)
        if err is not AxfcError.SUCCESS:
            logging.warning("IR build error: %s", err)
            return err, None

        # perform the translation from AIXIR to AIXGraph
        err, aix_graphs = self.__ir_translator.emit_aixh_graphs(aix_ir)

        if err is not AxfcError.SUCCESS:
            logging.warning("IR-to-AIXGraph translation error: %s", err)
            return err, None

        return AxfcError.SUCCESS, aix_graphs

    ## This method is used to dump out the generated AIXGraphs.
    #
    # @param self this object
    # @param out_path a file path to output the AIXGraphs
    # @param aix_graphs a list of AIXGraphs to be dumped out
    # @return error info
    def dump_aix_graphs(self, out_path: str, aix_graphs: list) -> AxfcError:
        logging.info("AxfcIRTranslator:dump_aix_graphs - %s", out_path)
        if aix_graphs is None:
            logging.warning("No AIXGraphs found")
            return AxfcError.INVALID_AIX_GRAPH

        # print out the generated AIX graphs
        for i, aix_graph in enumerate(aix_graphs):

            # rename the output path using the graph index
            tmp_path = out_path + ".%02d"%i

            # dump out each graph
            fd = open(tmp_path, mode="wt")
            fd.seek(0)
            fd.write(str(aix_graph))
            fd.close()

        return AxfcError.SUCCESS

    def dump_launcher(self, path: str) -> AxfcError:
        pass

    ## For debugging
    def __str__(self):
        pass