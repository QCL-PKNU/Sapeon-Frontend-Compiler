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

from AxfcError import *
from AxfcFrontendCompiler import *

def main():
    lg_path = "../tst/output.log"
    in_path = "../tst/mobilenet_v1_1.0_224_frozen.pb"
    md_path = "../src/aix_tf.md"

    os.remove(lg_path)
    logging.basicConfig(filename=lg_path, level=logging.INFO)

    logging.info("##########################")
    logging.info("# Start to compile")
    logging.info("##########################")

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
    main()
    pass