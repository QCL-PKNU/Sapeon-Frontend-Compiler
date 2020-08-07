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
    lg_path = args.log_path

    # for logging
    if lg_path is not None and os.path.isfile(lg_path):
        os.remove(lg_path)

    logging.basicConfig(filename=lg_path, level=logging.INFO)

    # for validating input files
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
        print("Error] Read machine description: ", err)
        return err

    err, aix_graph = fc.compile(in_path)
    if err is not AxfcError.SUCCESS:
        print("Error] Compile TF graph to AXIGraph: ", err)
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
    parser.add_argument('-l', '--log-path', metavar='', type=str, required=False,
                        help='path to log out')

    args = parser.parse_args()
    __main(args)
    pass