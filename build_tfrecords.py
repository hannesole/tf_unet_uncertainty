# ... for testing code snippets ...
from __future__ import division, print_function

import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import argparse

from timeit import default_timer as timer
# check how long tensorflow import takes
print('Importing tensorflow ...')
start = timer()
import tensorflow as tf
end = timer()
print('Elapsed time: %.4f' % (end - start))

import logging
from helper import std_image_util
from helper import std_logging
from helper.unet_hh import unet
from helper.unet_hh import util
#from tf_unet import util
# from tf_unet import image_gen
# from tf_unet import image_util
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from util import data_tfrecord
from util import data_hdf5
from unet import model

# SETTINGS
# -----------------
std_logging.initDebugLogger(os.path.join(sys.path[0], "log"), script_name="snippet_")

######################
# commandline arguments parser
# output folder is required, others optional
parser = argparse.ArgumentParser(description='Create dataset for tensorflow')
parser.add_argument('--name', '-n', metavar='dataset_name', help='The name of the resulting dataset (if not specified using time)',
                    default=time.strftime("%Y-%m-%d_%H%M"),  required=False)
# for testing TFRecord Data layer
parser.add_argument('--output_path', '-o', metavar='output_path', help='TFRecord dataset and other output is written to \"output_path/name/.\"',
                    default='/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/tfrecord/', required=False)
parser.add_argument('--dataset', '-d', metavar='dataset', help='The directory from which input images are read',
                    default="/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/train/std_v0_2_rgb/train/*.tif", required=False)
#"/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/train/std_v0_0/train/*.tif"
#parser.add_argument('--dataset', '-d', metavar='dataset_name', help='Either MosegChairs120 or MosegSintel (default chairs)', choices=['MosegChairs120', 'MosegSintel'], default='MosegChairs120', required=False)
args = parser.parse_args()
######################
# OVERRIDE ARGS

# custom changes to name
args.name = '1024x1024_rgbi'
mode='train'        # only needed for tfrecords
shape_img=[1024, 1024, 3]
shape_label=[1024, 1024, 1]
#args.name = args.name + '_256x256'
#args.name = '1024x1024_rgb'
#args.name = '256x256_rgb'

######################

# change tf_config for dacky (taken from flow net code)
config = tf.ConfigProto(log_device_placement=False)
if 'dacky' in os.uname()[1]:
    logging.info('Dacky: Running with memory usage limits')
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

# create output dir with name / current time (append _X if already exists)
out_dir = os.path.join(args.output_path, args.name)
if os.path.exists(out_dir):
    out_dir = out_dir + '_'
    num = 2
    while os.path.exists(out_dir + str(num)):
        num = num + 1
    out_dir = out_dir + str(num)

os.mkdir(out_dir)
logging.info('Writing output to out_dir: ' + out_dir)

sample_dir = out_dir + os.sep + 'samples'


# BUILD AND TEST TFRECORDS / DATALAYER
# ------------------------------------
# logging.debug('##################################################')
# logging.debug('# Building tfRecords, from: %s \n to: %s #' % (args.dataset, out_dir))
# logging.debug('##################################################')

# shape_img, shape_label = data_tfrecord.build_tf_records(args.dataset, out_dir)
#shape_img, shape_label = data_tfrecord.build_tf_records(args.dataset, out_dir, resize_shape=[256, 256])

# logging.debug('##################################################')
# logging.debug('# Built tfRecords, returned  %s %s #' % (str(shape_img), str(shape_label)))
# logging.debug('##################################################')

# data_path = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/tfrecord/2018-03-08_1821_2/train.tfrecords'
# shape_img = [1024, 1024, 4]
# shape_label = [1024, 1024, 1]
# # #out_dir = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/tfrecord/2018-03-04_1940/train.tfrecords'
# out_dir = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/tfrecord/2018-03-04_2029'

# data_tfrecord.test_tf_records(out_dir, mode='train', shape_img=shape_img, shape_label=shape_label, output_dir=sample_dir, batch_size=4)


# MERGE AND TEST HDF5Layer / DATALAYER
# ------------------------------------
args.dataset = 'data/hdf5/std_data_v0_2_pdf/train'
# logging.debug('##################################################')
# logging.debug('# MERGING HDF5 DATASETS from: %s \n to: %s #' % (args.dataset, args.dataset + os.sep + 'merged'))
# logging.debug('##################################################')

#data_hdf5.merge_hdf5(args.dataset, dest_folder=args.dataset + os.sep + 'merged', dest_filename='train.h5')
dest_filename = 'train_dset_chunked.h5'
dest_folder = args.dataset + os.sep + 'merged'
hdf5_dataset = dest_folder + os.sep + dest_filename
#hdf5_dataset = data_hdf5.merge_hdf5(args.dataset, dest_folder=args.dataset + os.sep + 'merged', dest_filename=dest_filename, chunks=True)

logging.debug('##################################################')
logging.debug('# TESTING HDF5 LAYER  #')
logging.debug('##################################################')

n_samples=10
batch_size = 3

data_hdf5.test_hdf5(data_path=hdf5_dataset, batch_size=batch_size, table_access=False)

