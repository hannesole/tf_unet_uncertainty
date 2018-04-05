# LOGGING FUNCTIONS
# =================
# These functions serve to create a log file when running a script.
# Apparently in the python configurations using virtual-env (pew) the root-logger is already configured somewhere.
# Therefore the command logging.basicConfig(...) doesn't work and a file logging handler has to be configured explicitly.
# Simply run the initDebugLogger() function to set a DEBUG logger that creates a file.
# Original (replaced) command was:
#
# logging.basicConfig(filename=os.path.join(sys.path[0], "output" , (time.strftime("%Y-%m-%d_%H%M") + '.log')),
#                     format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
#
# https://docs.python.org/3.5/howto/logging.html: format='%(asctime)s%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory-with-python
#
# Author: Hannes Horneber
# Date: 2018-03-18

import logging
import os
import time


def initDebugLogger(output_dir, script_name=""):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, (script_name + time.strftime("%Y-%m-%d_%H%M") + '.log')), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def initErrorLogger(output_dir, script_name=""):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, (script_name + time.strftime("%Y-%m-%d_%H%M") + '_err.log')), "w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def initConsoleLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #create console handler and set level to info
    #this is probably redundant since the root logger already displays messages at %(asctime)s :%(message)s"
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)