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

import argparse

from AxfcFrontendCompiler import *


## This is a main function for SKT-AIX frontend compiler
## @param params input parameters for the compilation
def __main(params):
    md_path = params.md_path
    in_path = params.in_path
    gv_path = params.graph_path
    log_path = params.log_path
    out_path = params.out_path
    cal_path = params.calib_path
    kernel_path = params.kernel
    aix_graph_format = params.aix_graph_format

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
    logging.info("# Start Compilation")
    logging.info("##########################")

    fc = AxfcFrontendCompiler()
    
    # read a machine description file
    err = fc.read_md_file(md_path)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Read machine description: %s", err)
        return err

    # read a calibration data file
    err = fc.read_calib_file(cal_path)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Read calibration data: %s", err)
        return err

    # perform the compilation
    err, aix_graphs = fc.compile(in_path)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Compile TF graph to AXIGraph: %s", err)
        return err

    # set the default output path if the path was not given
    if out_path is None:
        out_path = os.path.dirname(in_path) + "/aix_graph.out"

    # AIX graph out
    err = fc.dump_aix_graphs(out_path, aix_graphs, aix_graph_format)
    if err is not AxfcError.SUCCESS:
        logging.error("Error] Dump out AIXGraphs: %s", err)
        return err

    # AIX custom model
    dir = os.path.dirname(os.path.realpath('__file__'))
    tst_path = dir + '/tst'
    aix_graph_path = tst_path + '/aix_graph.out.'
    print(aix_graph_path)
    
    # err, custom_model_path = fc.dump_custom_model(path=in_path,
    #                            kernel_path=kernel_path,
    #                            aix_graph_path=aix_graph_path,
    #                            save_path=os.path.dirname(in_path))

    if err is not AxfcError.SUCCESS:
        logging.error("Error] Dump out AIX custom model: %s", err)

    # AIX Launcher: TESTING
    # if kernel_path is not None:
    #
    #     custom_model_path = '../tst/custom_model.pb'
    #     output = fc.dump_launcher(custom_model_path=custom_model_path,
    #                               kernel_op_path=kernel_path)
    #     print('Prediction:')
    #     print(output)

    # for AIXIR graph viewer
    # if gv_path is not None:
    #     aix_ir_graph = fc.get_ir_graph()
    #     err = aix_ir_graph.dump_to_file(gv_path, ["Const", "Pad"])
    #     if err is not AxfcError.SUCCESS:
    #         logging.error("Error] Dump out AIXIR graph: %s", err)
    #         return err

    logging.info("##########################")
    logging.info("# Finished Compilation")
    logging.info("##########################")

    return AxfcError.SUCCESS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SKT AIX Frontend Compiler',
        usage='use "%(prog)s -h/--help" for more information')

    parser.add_argument('-m', '--md-path', metavar='', type=str, required=True,
                        help='path to a machine description file')
    parser.add_argument('-i', '--in-path', metavar='', type=str, required=True,
                        help='path to the protocol buffer of a frozen model')
    parser.add_argument('-k', '--kernel', metavar='', type=str, required=False,
                        help='path to the kernel (custom operation kernel *.so) file')
    parser.add_argument('-c', '--calib-path', metavar='', type=str, required=False,
                        help='path to the calibration data of a frozen model')
    parser.add_argument('-o', '--out-path', metavar='', type=str, required=False,
                        help='path to output the generated AIXGraph')
    parser.add_argument('-l', '--log-path', metavar='', type=str, required=False,
                        help='path to log out')
    parser.add_argument('-g', '--graph-path', metavar='', type=str, required=False,
                        help='path to dump an IR graph')
    parser.add_argument('-f', '--aix_graph_format', metavar='', type=str, required=False,
                        help='configure output format between binary and text for aix graph')

    args = parser.parse_args()
    __main(args)

    pass
