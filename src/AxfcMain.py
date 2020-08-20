#######################################################################
#   AxfcMain
#
#   Created: 2020. 08. 03
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import os
import logging
import argparse

from AxfcError import *
from AxfcFrontendCompiler import *

def __main(args):
    md_path = args.md_path
    in_path = args.in_path
    gv_path = args.graph_path
    log_path = args.log_path
    out_path = args.out_path

    # for logging
    if log_path is not None and os.path.isfile(log_path):
        os.remove(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO)

    # for validating md and input files
    if not os.path.isfile(md_path):
        print("Invalid path to an MD file: " + md_path)
        return

    if not os.path.isfile(in_path):
        print("Invalid path to an input frozen model: " + in_path)
        return

    logging.info("##########################")
    logging.info("# Start to compile")
    logging.info("##########################")

    # for compilation
    fc = AxfcFrontendCompiler()

    err = fc.read_md_file(md_path)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Read machine description: %s", err)
        return err

    err, aix_graphs = fc.compile(in_path)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Compile TF graph to AXIGraph: %s", err)
        return err

    # set the default output path if the path was not given
    if out_path is None:
        out_path = os.path.dirname(in_path) + "/aix_graph.out"

    # AIX graph out
    err = fc.dump_aix_graphs(out_path, aix_graphs)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Dump out AIXGraphs: %s", err)
        return err

    # for AIXIR graph viewer
    if gv_path is not None:
        aix_ir_graph = fc.get_ir_graph()
        err = aix_ir_graph.dump_to_file(gv_path, ["Const", "Pad"])
        if err is not AxfcError.SUCCESS:
            logging.error("Error] Dump out AIXIR graph: %s", err)
            return err

    logging.info("##########################")
    logging.info("# Finish to compile")
    logging.info("##########################")

    return AxfcError.SUCCESS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='SKT AIX Frontend Compiler',
            usage='use "%(prog)s -h/--help" for more information')

    parser.add_argument('-m', '--md-path',  metavar='', type=str, required=True,
                        help='path to a machine description file')
    parser.add_argument('-i', '--in-path',  metavar='', type=str, required=True,
                        help='path to the protocol buffer of a frozen model')
    parser.add_argument('-o', '--out-path', metavar='', type=str, required=False,
                        help='path to output the generated AIXGraph')
    parser.add_argument('-l', '--log-path', metavar='', type=str, required=False,
                        help='path to log out')
    parser.add_argument('-g', '--graph-path', metavar='', type=str, required=False,
                        help='path to dump an IR graph')

    args = parser.parse_args()
    __main(args)

    pass
