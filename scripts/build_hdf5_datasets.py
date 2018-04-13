# HDF5 split into sets
# =========================
#
# Was used to tile the hdf5 dataset into smaller pieces
# to create a train / test and validation set
#
# Author: Hannes Horneber
# Date: 2018-04-13


import logging
from util import logs
from util import data_hdf5

# console logger - depending on your settings this might not be necessary and produce doubled output
logs.initConsoleLogger()


hdf5_folder = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/hdf5/'
datasources = ['/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/' +
               'data/hdf5/std_data_v0_2_merged.h5']

data_hdf5.split_into_sets(hdf5_folder = hdf5_folder, dest_folder=hdf5_folder, datasources=datasources,
                     element_axis=0, validate=False,
                     chunks=None)
