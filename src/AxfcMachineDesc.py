#######################################################################
#   AxfcMachineDesc
#
#   Created: 2020. 08. 03

#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   Copyright (c) 2020
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import json
import logging

from AxfcError import *

##
# AxfcMachineDesc
#
from typing import Dict


class AxfcMachineDesc:

    # for input type
    TYPE_TENSORFLOW = 0
    TYPE_PYTORCH = 1
    TYPE_MXNET = 2
    TYPE_UNKNOWN = 3

    # for hardware acceleration
    DEFAULT_PROFIT_THRESHOLD = 1000

    ## @var __info
    # general info. of dictionary type for AIX compiler

    ## @var __aix_op
    # general machine description info. of dictionary type

    ## The constructor
    def __init__(self):
        self.__info = dict()
        self.__aix_ops = dict()

    ## This method returns the type of AI framework.
    #

    ## This method is used to read a machine description from the given input path.
    # @param self this object
    # @param path file path of the machine description
    # @return error info
    def read_file(self, path: str) -> AxfcError:
        logging.info("AxfcMachineDesc:read_file - path: %s", path)

        # read machine description
        try:
            with open(path, "rt") as md_file:
                self.__info = json.load(md_file)
        except FileNotFoundError:
            return AxfcError.INVALID_FILE_PATH
        except ValueError:
            return AxfcError.INVALID_MD_FORMAT

        # make a dictionary of AIX operations
        try:
            self.__aix_ops = self.__info["AIX_OPERATION"]
        except ValueError:
            return AxfcError.INVALID_MD_FORMAT

        return AxfcError.SUCCESS

    ## This method indicates whether the given operation is supported by the AIX hardware or not.
    # @param self this object
    # @param op_name operation name
    # @return the input type of the frontend compilation
    def get_axih_support(self, op_name: str) -> bool:
        #logging.info("AxfcMachineDesc:check_axih_support - %s", op_name)
        try:
            if self.__aix_ops[op_name] is not None:
                return True
        except KeyError:
            return False

    ## This method returns the input type of the frontend compilation.
    # @param self this object
    # @return the input type of the frontend compilation
    def get_in_type(self):
        #logging.info("AxfcMachineDesc:get_in_type")
        try:
            in_type = self.__info['AIX_INPUT_TYPE']

            if in_type == "TENSORFLOW":
                return AxfcMachineDesc.TYPE_TENSORFLOW
            else:
                return AxfcMachineDesc.TYPE_UNKNOWN
        except KeyError:
            logging.warning("AxfcMachineDesc:get_in_type - KeyError occurs")
            return AxfcMachineDesc.TYPE_UNKNOWN

    ## This method returns the input type of the frontend compilation.
    # @param self this object
    # @return the profit threshold to determine whether to use hardware acceleration
    def get_profit_threshold(self):
        #logging.info("AxfcMachineDesc:get_profit_threshold")
        try:
            threshold = self.__info['AIX_PROFIT_THRESHOLD']
            return int(threshold)
        except KeyError:
            logging.warning("AxfcMachineDesc:get_profit_threshold - KeyError occurs")
            return DEFAULT_PROFIT_THRESHOLD

    ## For debugging
    def __str__(self):
        str_buf = ">> Machine Description: \n" + str(self.__info)
        str_buf = ">> AIX Operations: \n" + str(self.__aix_ops)
        return str_buf + "\n"