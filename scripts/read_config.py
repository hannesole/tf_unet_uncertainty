# READ CONFIG FILE
# =========================
#
# Example and test code to read a config file.
#
# Author: Hannes Horneber
# Date: 2018-04-10

import configparser
import logging
import os, sys
from util import logs
from collections import OrderedDict
from util import config_util

# ######################################################################################################################
# LOGGING OPTIONS
# ---------------
# console logger - depending on your settings this might not be necessary and produce doubled output
logs.initConsoleLogger()
# ######################################################################################################################

config = configparser.ConfigParser()
config.read('config.ini')

class config_extended(config_util.config_decorator):
    # wrapper class for all options (similar to a C++ struct)
    # ---------> Training
    train = True
    # ---------> Testing
    test = True

opts = config_extended(config['DEFAULT'])
opts_test = config_util.config_decorator(config['TEST'])
opts_train = config_util.config_decorator(config['TRAIN'])


logging.info(config.sections())
logging.info(OrderedDict(config.items('TEST')))
dict(config.items('TRAIN'))
# parsed_vals = OrderedDict([(key, options.config_decorator.parse_implicitly(config['TEST'][key]))
#                            for key in OrderedDict(config.items('TEST')) ])
logging.info(opts.get_attr_val_list())
logging.info(opts_test.get_attr_val_list())
logging.info(opts_train.get_attr_val_list())

opts = config_util.config_decorator(config['DEFAULT'])

logging.info(opts.resample_n)
logging.info(opts.resample_n)
logging.info(opts.resample_n)