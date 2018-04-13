# UNET TRAINING AND TESTING
# =========================
#
# Python (commandline) script for training and/or testing a Unet.
#
# Author: Hannes Horneber
# Date: 2018-03-18

from __future__ import division, print_function

import os
import sys
import argparse
import shutil
import time
import datetime
import matplotlib; matplotlib.use('Agg') # set this before any other module makes use of matplotlib (and sets it)
import numpy as np
from timeit import default_timer as timer

# check how long tensorflow import takes
print('Importing tensorflow ...')
t_start = timer()
import tensorflow as tf
print('Elapsed time: %.4f s' % (timer() - t_start))
from tensorflow.python import debug as tf_debug
from tensorflow import contrib as tc

import logging
from util import img_util
from util import calc
from util import logs
#from unet import model
from unet import model_dev as model
from unet import data_layers
from util.tfutils import SimpleTrainer

# ######################################################################################################################
# LOGGING OPTIONS
# ---------------
# output log to output folder [script-location]/log
logs.initDebugLogger(os.path.join(sys.path[0], "log"), script_name="std_unet")
# output error log to output folder [script-location]/log
logs.initErrorLogger(os.path.join(sys.path[0], "log"), script_name="std_unet-err")
# console logger - depending on your settings this might not be necessary and produce doubled output
logs.initConsoleLogger()

# ######################################################################################################################
# COMMANDLINE ARGUMENTS PARSER
# ----------------------------
# all required are set to False to be able to run this script from environments w/o commandline options
# change defaults or override commandline args below to change what you are providing

parser = argparse.ArgumentParser(description='UNet Training and Testing')
# TRAINING & TESTING arguments
parser.add_argument('--dataset', '-d', metavar='dataset', required=False,
                    help='Path to the dataset that will be used for training/testing',
                    default="/home/hornebeh/proj_tf_unet/data/tfrecord/1024x1024_rgbi/train.tfrecords")
parser.add_argument('--checkpoint', '-c', metavar='checkpoint', required=False,
                    help='Provide a model checkpoint file, otherwise searching for last one in train_dir/checkpoints.' +
                        'For training: Trains from scratch if no checkpoint is found.' +
                        'For testing: Breaks if no checkpoint is found (model cannot initialize).',
                    default=None)
parser.add_argument('--mode', '-m', metavar='mode', required=False,
                    help="CL parameter to activate phases ['train' or 'test'] or " +
                         "or modes ['tfdbg': TensorFlow CLI debugger, 'debug': Additonal debug code]. " +
                         "To activate multiple phases, just concatenate strings in arbitrary order, " +
                         "e.g. ('traintest' or 'testtrain' or 'debugtrain').",
                    default=None)

#   for output of training / input of testing:
#       either output_dir/name ...
parser.add_argument('--name', '-n', metavar='train_name', required=False,
                    help='The training session name (if not specified using time only)',
                    default=time.strftime("%Y-%m-%d_%H%M"))
parser.add_argument('--output_dir', '-o', metavar='output_dir', required=False,
                    help='Model and training files are written to train_dir=\"output_dir/name/\".',
                    default='/home/hornebeh/proj_tf_unet/output/')
#       ... or a full train_dir path:
parser.add_argument('--train_dir', '-t', metavar='train_dir', required=False,
                    help='Directory where models (checkpoints) are stored during training and loaded for testing. ' +
                         'Usually generated as train_dir=\"output_dir/name/\".' +
                         'Pass this if you don\'t want to specify name and dir separately ' +
                         '(e.g. for testing only an already trained network) - or if for some reason you want to use a ' +
                         'different directory than the one that is by default generated for and used during training ' +
                         '(this overrides output_dir and name).',
                    default=None)

# TESTING ARGUMENTS
parser.add_argument('--test_dir', '-p', metavar='test_dir', required=False,
                    help='Prediction is written to \"test_dir/\". ' +
                         'If not specified using prediction subfolder in train_dir.',
                    default=None)

args = parser.parse_args()


# ######################################################################################################################
# SCRIPT ARGS / OVERRIDE ARGS
# ---------------------------
def _________________________OPTIONS___________________________(): pass # dummy function for PyCharm IDE
# Since I usually call this from an IDE, not all script arguments are implemented as commandline arguments.
# You can also override commandline defaults here.
class opts(object): #TODO Outsource to opts / config file
    # wrapper class for all options (similar to a C++ struct)
    # -> General
    debug = True if args.mode is None else ('debug' in args.mode)
    tfdbg = False if args.mode is None else ('tfdbg' in args.mode)
    copy_script_dir = 'train_dir'

    # ---------> Training & Testing
    shuffle = True  # shuffle data input
    augment = True  # randomly augment data input
    resample_n_train = None  # resample same image to use always the same image in training
    batch_size = 1  # simultaneous processing of batch_size images (GPU mem is limit)

    # working configurations for 12 GB GPU RAM:
    # size 512x512: batch_size= 1 | 2 | 8 ; n_start_features= 64 | 64 | 16
    n_contracting_blocks = 5
    n_start_features = 64
    resize = [512, 512] # None
    resize_method = None # "scale" or "random_crop" (default)

    norm_fn = None # tc.layers.batch_norm  #
    normalizer_params = {'is_training': None,   # set to None if it should correspond to Phase
                         'decay': 0.9,          # more stable than default (0.999)
                         'zero_debias_moving_mean': True } # for stability

    data_layer_type = 'feed_dict' #   'hdf5'  # see unet model / data_layers for specifics
    shape_img = [1024, 1024, 4]  # of a single image
    shape_label = [1024, 1024, 1]  # of a single label mask
    shape_weights = [1024, 1024, 1]  # of a single weight mask

    # ---------> Training
    train = True if args.mode is None else ('train' in args.mode)  # script will train
    train_name = 'augment_saver'

    # ---------> Testing
    test = True if args.mode is None else ('test' in args.mode)  # script will do testing
    n_samples = 200       # ... of n_samples
    resample_n = 20    # resample same image to calculate uncertainty, N or None
    if resample_n is not None: resize_method = "center_crop"
    keep_prob_test = 0.5 # keep_prob  # use 1.0 for no dropout during test-time
    pass

# -> Override CLI arguments (for convenience when running from IDE)
PROJECT_DIR = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/'

#args.output_dir = '/home/hornebeh/proj_tf_unet/output/'
args.name = 'unet_' + time.strftime("%Y-%m-%d_%H%M") + '_sh-aug-kp09-60k'
#args.name = 'overwrite'

#args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-04-05_1409_debug_batch_norm'
#args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-04-05_1425_debug_no_batch_norm'
args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-04-05_1713_sh-aug-kp1-60k'
args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-04-05_1717_sh-aug-kp09-60k'

#args.train_dir = PROJECT_DIR + 'output_scr/' + 'unet_2018-03-25_2021_augment_saver'
#args.train_dir = PROJECT_DIR + 'output_scr/' + 'unet_2018-03-28_1813_debug_no_droput'

#args.checkpoint = PROJECT_DIR + 'output_scr/' + unet_2018-03-22_2021_augment/checkpoints/snapshot-34000"
#args.checkpoint = PROJECT_DIR + 'output/' + 'unet_2018-04-05_1425_debug_no_batch_norm/checkpoints/snapshot-4000'


opts.data_layer_type = 'hdf5'
args.dataset = PROJECT_DIR + "data/hdf5/std_data_v0_2_pdf/train/merged/train_dset_chunked.h5"
opts.shape_img = [1024, 1024, 4]
opts.shape_label = [1024, 1024, 1]
opts.shape_weights = [1024, 1024, 1]

# opts.data_layer_type = 'tfrecords'
# args.dataset = "/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/"+ \
#                "data/tfrecord/1024x1024_rgbi/test.tfrecords"
# opts.shape_img = [1024, 1024, 4]
# opts.shape_label = [1024, 1024, 1]

# args.dataset = "/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/"+ \
#                "data/tfrecord/1024x1024_rgb/train.tfrecords"
# opts.shape_img = [1024, 1024, 3]
# opts.shape_label = [1024, 1024, 1]

# args.dataset = "/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/"+ \
#                "data/tfrecord/256x256_rgb/train.tfrecords"
# opts.shape_img = [256, 256, 3]
# opts.shape_label = [256, 256, 1]

def opts_to_str(opts):
    '''Prints the opts class as box with all settings'''
    # vars(opts) == opts.__dict__  # but in different order!
    opts_dict =  sorted(opts.__dict__.items())
    my_opts = [(opt, str(opts.__dict__[opt])) for opt in opts.__dict__ if not opt.startswith('__')]
    my_opts = sorted(my_opts, key = lambda x: x[0])
    opts_width = 64
    opts_str = '_' * opts_width + '\n| SCRIPT OPTIONS:' + ' ' * (opts_width - 17 - 1) + '|'
    for opt in my_opts:
        opts_str = opts_str + '\n|   ' + opt[0] + '.' * (opts_width // 2 - len(opt[0]) - 5) + ':  ' \
                   + opt[1] + ' ' * (opts_width // 2 - len(opt[1]) - 3) + '|'
    opts_str = opts_str + '\n' + '_' * opts_width + '\n'
    return opts_str

# ######################################################################################################################
# SETUP VARS AND DIRS
# -------------------

# if opts.debug: logging.debug(str(os.environ))    # log environment (to check whether CUDA paths are set correctly etc.)
if opts.debug: logging.debug('os.uname: %s' % (str(os.uname())))  # log uname to check which node code is running on

tf_config = tf.ConfigProto(log_device_placement=False)
if 'dacky' in os.uname()[1]:
    logging.info('Dacky: Running with memory usage limits')
    # change tf_config for dacky to use only 1 GPU
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    # change tf_config for lmb_cluster so that GPU is visible and utilized
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# create train directory train_dir if it doesn't exist
if args.train_dir is None:  # use default naming if no parameter is passed for train_dir
    # create train_dir with name (by default current time)
    train_dir = os.path.join(args.output_dir, args.name)
    # avoid overwriting (by appendin _X to name if already exists)
    if os.path.exists(train_dir):
        if not args.name == 'overwrite':
            train_dir = train_dir + '_'
            num = 2
            while os.path.exists(train_dir + str(num)):
                num = num + 1
            train_dir = train_dir + str(num)
        else:
            if opts.train: shutil.rmtree(train_dir)
    if not os.path.exists(train_dir): os.mkdir(train_dir)
    logging.info('For writing model and other output, created train_dir %s ' % train_dir)
else:
    # make sure train_dir exists
    train_dir = args.train_dir
    if not os.path.exists(train_dir):
        logging.info('For writing model and other output, created train_dir %s ' % train_dir)
        os.mkdir(train_dir)
    else:
        logging.info('For writing model and other output, using existing train_dir %s ' % train_dir)

# create test_dir if it doesn't exist. Use either default naming or a given parameter.
if args.predict_dir is None:
    # create test_dir with current time if none is specified
    args.predict_dir = train_dir + os.sep + "prediction" + '_' + time.strftime("%Y-%m-%d_%H%M")
    # avoid overwriting (by appendin _X to name if already exists)
    if os.path.exists(args.predict_dir):
        args.predict_dir = args.predict_dir + '_'
        num = 2
        while os.path.exists(args.predict_dir + str(num)):
            num = num + 1
        args.predict_dir = args.predict_dir + str(num)
    os.mkdir(args.predict_dir)
else:
    # make sure test_dir exists
    if not os.path.exists(args.predict_dir):
        os.mkdir(args.predict_dir)

# copy script to train_dir
if opts.copy_script_dir is not None:
    if opts.copy_script_dir == 'train_dir':
        opts.copy_script_dir = train_dir + os.sep + 'py_src'
    if not os.path.exists(opts.copy_script_dir):
        os.mkdir(opts.copy_script_dir)
    copy_file_path = opts.copy_script_dir + os.sep + os.path.basename(__file__)
    if os.path.exists(copy_file_path): copy_file_path = copy_file_path + '_' + time.strftime("%Y-%m-%d_%H%M")
    shutil.copy(__file__, copy_file_path)
    logging.debug('Copied py source to: ' + copy_file_path)


print(opts_to_str(opts))

# ######################################################################################################################
# SETUP FUNCTIONS FOR TESTING AND TRAINING
# ----------------------------------------
def ______________________________________________________(): pass # dummy function for PyCharm IDE

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    logging.debug('> initialize_uninitialized. Found %s ' % str(len(not_initialized_vars)))
    if len(not_initialized_vars):
        print('\n[...]\n'.join([str(not_initialized_vars[i].name) for i in [0, len(not_initialized_vars) - 1]]))
        sess.run(tf.variables_initializer(not_initialized_vars))


data_dir = "/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/data/train/std_v0_2_rgb/train/*.tif"
data_dir = "/misc/lmbraid12/hornebeh/dd/data/tif/tiles/retileALL/*.tif"
#def get_test_data(data_dir):

from util import data_import


# ######################################################################################################################
# TESTING
# -------
def ___________________________TEST______________________________(): pass # dummy function for PyCharm IDE
if opts.train: tf.reset_default_graph()

# core code for testing
def test_feed(sess, net_test):
    logging.info('#-----------------------------------------------#')
    logging.info('#               Starting Testing                #')
    logging.info('#-----------------------------------------------#')

    # load model for testing (if None provided, searches in train_dir, if not found doesn't load)
    trainer = SimpleTrainer(session=sess, train_dir=train_dir)
    chkpt_loaded = trainer.load_checkpoint(args.checkpoint)
    # init variables if no checkpoint was loaded
    if not chkpt_loaded: sess.run(tf.group(tf.global_variables_initializer()))
    logging.info("Loaded variables from checkpoint" if chkpt_loaded else "Randomly initialized (!) variables")

    # ###########################################################################
    # RUN UNET
    # ###########################################################################
    logging.debug("predicting, sampling %s times, batch_size %s" % (opts.n_samples, opts.batch_size))

    for b in range(opts.n_samples):
        data_provider = data_import.create_img_batch_provider(data_dir, opts.batch_size, shuffle_data=True)
        batch_img = next(data_provider)
        logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))

        batch_activations, batch_prediction = sess.run([net_test.output_mask, net_test.prediction],
                                                       feed_dict={net_test.batch_img: batch_img})

        # out_img = np.squeeze(img_util.to_rgb(batch_activations))
        # img_util.save_image(out_img, "%s/img_%s_pred.png" % (args.test_dir, b))

        logging.debug('batch_activations: %s %s' % (str(batch_activations.shape), str(batch_activations.dtype)))
        logging.debug('batch_prediction: %s %s' % (str(batch_prediction.shape), str(batch_prediction.dtype)))

        r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
        r_batch_img = r_batch_img[...,0:3] # ignore fourth channel
        r_batch_activations = np.reshape(batch_activations,
                                         [-1, batch_activations.shape[2], batch_activations.shape[3]])
        r_batch_prediction = np.reshape(batch_prediction, [-1, batch_prediction.shape[2]])

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 0, np.newaxis], normalize=True)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 1, np.newaxis], normalize=True)),
                                  np.squeeze(img_util.to_rgb(r_batch_prediction[..., np.newaxis]))
                                  ), axis=1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))


    # ###########################################################################
    # CLOSE NET
    # ###########################################################################


# test with sampling for uncertainty (only makes sense when resample_n != None and keep_prob != 1.0
def test_feed_sampling(sess, net_test):
    logging.info('#-----------------------------------------------#')
    logging.info('#        Starting Testing with sampling         #')
    logging.info('#-----------------------------------------------#')

    # # init variables
    # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    trainer = SimpleTrainer(session=sess, train_dir=train_dir)
    # load model (if None provided, gets latest from train_dir/checkpoints, if none found doesn't load)
    trainer.load_checkpoint(args.checkpoint)

    # ###########################################################################
    # RUN UNET
    # ###########################################################################
    logging.debug("predicting, sampling %s x %s times, batch_size %s" %
                  (str(opts.n_samples), str(opts.resample_n), str(opts.batch_size)))

    sample_dir  = args.predict_dir + os.sep + 'samples'
    os.mkdir(sample_dir)
    for b in range(opts.n_samples):
        data_provider = data_import.create_img_batch_provider(data_dir, opts.batch_size, shuffle_data=True)
        batch_img = next(data_provider)
        logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))
        if opts.resize_method == "center_crop":
            batch_img = img_util.crop_to_shape(batch_img, [batch_img.shape[0], opts.resize[0], opts.resize[1], batch_img.shape[3]])
            logging.debug(' resized to: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))

        for s in range(opts.resample_n):
            batch_activations, batch_pred = sess.run([net_test.output_mask, net_test.prediction],
                                                           feed_dict={net_test.batch_img: batch_img})
            logging.debug('prediction: %s %s' % (str(batch_pred.shape), str(batch_pred.dtype)))

            r_batch_pred = np.reshape(batch_pred, [-1, batch_pred.shape[2]])
            if s == 0: prediction_samples = np.zeros([opts.resample_n] + list(r_batch_pred.shape), dtype=np.uint8)
            prediction_samples[s, ...] = r_batch_pred

            out_sample = img_util.to_rgb(prediction_samples[s, ...])
            img_util.save_image(out_sample, "%s/sample_%s_%s.png" % (sample_dir, b, s))

        logging.info('finished resampling (%s), calculating entropy' % (str(opts.resample_n)))

        entropy = calc.entropy_bin_array(prediction_samples)
        mean = np.mean(prediction_samples, axis=0)
        std = np.std(prediction_samples, axis=0)

        r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
        #r_batch_img_rgb = r_batch_img[..., 0:3]  # ignore fourth channel
        # r_batch_img_new[..., 0] = r_batch_img[..., 1]
        # r_batch_img_new[..., 1] = r_batch_img[..., 0]
        # r_batch_img_new[..., 2] = r_batch_img[..., 2]

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img_rgb)),
                                  np.squeeze(
                                      img_util.to_rgb(mean)),
                                  np.squeeze(
                                      img_util.to_rgb_heatmap(entropy, rgb_256=True)),
                                  np.squeeze(
                                      img_util.to_rgb_heatmap(std, rgb_256=True))
                                  ), axis=1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))

    # ###########################################################################
    # CLOSE NET
    # ###########################################################################




# core code for testing
def test_core(sess, net_test):
    logging.info('#-----------------------------------------------#')
    logging.info('#               Starting Testing                #')
    logging.info('#-----------------------------------------------#')

    # load model for testing (if None provided, searches in train_dir, if not found doesn't load)
    trainer = SimpleTrainer(session=sess, train_dir=train_dir)
    chkpt_loaded = trainer.load_checkpoint(args.checkpoint)
    # init variables if no checkpoint was loaded
    if not chkpt_loaded: sess.run(tf.group(tf.global_variables_initializer()))
    logging.info("Loaded variables from checkpoint" if chkpt_loaded else "Randomly initialized (!) variables")

    # ###########################################################################
    # RUN UNET
    # ###########################################################################
    logging.debug("predicting, sampling %s times, batch_size %s" % (opts.n_samples, opts.batch_size))

    for b in range(opts.n_samples):
        batch_img, batch_label, batch_activations, batch_prediction = sess.run(
            [net_test.batch_img, net_test.batch_label, net_test.output_mask, net_test.prediction])

        # out_img = np.squeeze(img_util.to_rgb(batch_activations))
        # img_util.save_image(out_img, "%s/img_%s_pred.png" % (args.test_dir, b))

        logging.debug('batch_activations: %s %s' % (str(batch_activations.shape), str(batch_activations.dtype)))
        logging.debug('batch_prediction: %s %s' % (str(batch_prediction.shape), str(batch_prediction.dtype)))
        logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))
        logging.debug('batch_label: %s %s' % (str(batch_label.shape), str(batch_label.dtype)))

        r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
        r_batch_label = np.reshape(batch_label, [-1, batch_label.shape[2], batch_label.shape[3]])
        r_batch_activations = np.reshape(batch_activations,
                                         [-1, batch_activations.shape[2], batch_activations.shape[3]])
        r_batch_prediction = np.reshape(batch_prediction, [-1, batch_prediction.shape[2]])

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                  np.squeeze(img_util.to_rgb(r_batch_label)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 0, np.newaxis], normalize=True)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 1, np.newaxis], normalize=True)),
                                  np.squeeze(img_util.to_rgb(r_batch_prediction[..., np.newaxis]))
                                  ), axis=1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))


    # ###########################################################################
    # CLOSE NET
    # ###########################################################################


def test_debug(sess, net_test):
    logging.info('#-----------------------------------------------#')
    logging.info('#               Starting Testing (debug)        #')
    logging.info('#-----------------------------------------------#')

    # load model for testing (if None provided, searches in train_dir, if not found doesn't load)
    logging.info("Attempt restore with SimpleTrainer load from: %s" % args.checkpoint)
    # load model for continued training (if None provided, searches in train_dir, if not found doesn't load)
    trainer = SimpleTrainer(session=sess, train_dir=train_dir)
    chkpt_loaded = trainer.load_checkpoint(args.checkpoint)

    # init variables if no checkpoint was loaded
    if not chkpt_loaded: sess.run(tf.group(tf.global_variables_initializer()))
    logging.info("Loaded variables from checkpoint" if chkpt_loaded else "Randomly initialized (!) variables")

    # in case any variables are not yet initialized
    #initialize_uninitialized(sess)

    # ###########################################################################
    # RUN UNET
    # ###########################################################################
    logging.debug("predicting, sampling %s times, batch_size %s" % (opts.n_samples, opts.batch_size))

    for b in range(opts.n_samples):
        batch_img, batch_label, batch_activations, batch_prediction = sess.run(
            [net_test.batch_img, net_test.batch_label, net_test.output_mask, net_test.prediction])

        # out_img = np.squeeze(img_util.to_rgb(batch_activations))
        # img_util.save_image(out_img, "%s/img_%s_pred.png" % (args.test_dir, b))

        logging.debug('batch_activations: %s %s' % (str(batch_activations.shape), str(batch_activations.dtype)))
        logging.debug('batch_prediction: %s %s' % (str(batch_prediction.shape), str(batch_prediction.dtype)))
        logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))
        logging.debug('batch_label: %s %s' % (str(batch_label.shape), str(batch_label.dtype)))

        # logging.debug('describe prediction_samples: ' + str(stats.describe(batch_activations)))
        # logging.debug('describe prediction_samples[0]: ' + str(stats.describe(prediction_samples[0])))
        # out_img = img_util.combine_img_prediction(batch_img, batch_label, batch_activations)

        r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
        r_batch_label = np.reshape(batch_label, [-1, batch_label.shape[2], batch_label.shape[3]])
        r_batch_activations = np.reshape(batch_activations, [-1, batch_activations.shape[2], batch_activations.shape[3]])
        r_batch_prediction = np.reshape(batch_prediction, [-1, batch_prediction.shape[2]])

        #r_batch_softmax = calc.softmax(r_batch_activations, axis=-1) # slow
        argmax = np.argmax(r_batch_activations, axis=-1) # just take direct max

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                  np.squeeze(img_util.to_rgb(r_batch_label)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 0, np.newaxis], normalize=False)),
                                  np.squeeze(img_util.to_rgb(r_batch_activations[..., 1, np.newaxis], normalize=False)),
                                  np.squeeze(img_util.to_rgb(argmax[..., np.newaxis])),
                                  np.squeeze(img_util.to_rgb(r_batch_prediction[..., np.newaxis]))
                                 ), axis = 1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))

    # ###########################################################################
    # CLOSE NET
    # ###########################################################################


# test with sampling for uncertainty (only makes sense when resample_n != None and keep_prob != 1.0
def test_sampling(sess, net_test):
    logging.info('#-----------------------------------------------#')
    logging.info('#        Starting Testing with sampling         #')
    logging.info('#-----------------------------------------------#')

    # # init variables
    # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    trainer = SimpleTrainer(session=sess, train_dir=train_dir)
    # load model (if None provided, gets latest from train_dir/checkpoints, if none found doesn't load)
    trainer.load_checkpoint(args.checkpoint)

    # ###########################################################################
    # RUN UNET
    # ###########################################################################
    logging.debug("predicting, sampling %s x %s times, batch_size %s" %
                  (str(opts.n_samples), str(opts.resample_n), str(opts.batch_size)))

    sample_dir  = args.predict_dir + os.sep + 'samples'
    os.mkdir(sample_dir)
    for b in range(opts.n_samples):

        for s in range(opts.resample_n):
            if s == 0:
                batch_img, batch_label, batch_pred = sess.run(
                    [net_test.batch_img, net_test.batch_label, net_test.prediction])
                logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))
                logging.debug('batch_label: %s %s' % (str(batch_label.shape), str(batch_label.dtype)))
                logging.debug('prediction: %s %s' % (str(batch_pred.shape), str(batch_pred.dtype)))
            else:
                _, _, batch_pred = sess.run(
                    [net_test.batch_img, net_test.batch_label, net_test.prediction])
                logging.debug('prediction: %s %s' % (str(prediction_samples[s, ...].shape), str(prediction_samples[s, ...].dtype)))

            r_batch_pred = np.reshape(batch_pred, [-1, batch_pred.shape[2]])
            if s == 0: prediction_samples = np.zeros([opts.resample_n] + list(r_batch_pred.shape), dtype=np.uint8)
            prediction_samples[s, ...] = r_batch_pred

            out_sample = img_util.to_rgb(prediction_samples[s, ...])
            img_util.save_image(out_sample, "%s/sample_%s_%s.png" % (sample_dir, b, s))

        logging.info('finished resampling (%s), calculating entropy' % (str(opts.resample_n)))

        entropy = calc.entropy_bin_array(prediction_samples)
        mean = np.mean(prediction_samples, axis=0)
        std = np.std(prediction_samples, axis=0)

        r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
        r_batch_label = np.reshape(batch_label, [-1, batch_label.shape[2], batch_label.shape[3]])

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                  np.squeeze(img_util.to_rgb(r_batch_label)),
                                  np.squeeze(
                                      img_util.to_rgb(mean)),
                                  np.squeeze(
                                      img_util.to_rgb_heatmap(entropy, rgb_256=True)),
                                  np.squeeze(
                                      img_util.to_rgb_heatmap(std, rgb_256=True))
                                  ), axis=1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))

    # ###########################################################################
    # CLOSE NET
    # ###########################################################################


def TEST(): pass
if opts.test:
    logging.info('####################################################################')
    logging.info('#                            TESTING                               #')
    logging.info('####################################################################')

    # create network graph with data layer
    net = model.UNet(dataset_pth=args.dataset,
                     shape_img=opts.shape_img, shape_label=opts.shape_label, shape_weights=opts.shape_weights,
                     batch_size=opts.batch_size, shuffle=False, augment=False,
                     resize=opts.resize, resize_method=opts.resize_method,
                     data_layer_type=opts.data_layer_type,
                     n_contracting_blocks=opts.n_contracting_blocks, n_start_features=opts.n_start_features,
                     norm_fn=opts.norm_fn, normalizer_params=opts.normalizer_params,
                     resample_n=opts.resample_n,
                     is_training=False, keep_prob=opts.keep_prob_test,
                     prefetch_n=None, prefetch_threads=None,
                     debug=opts.debug, copy_script_dir=None
                     )

    with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=tf_config)) if opts.tfdbg \
            else tf.Session(config=tf_config) as sess:
        if opts.debug:
            test_debug(sess, net)
        else:
            if opts.resample_n is not None: test_feed_sampling(sess, net)
            else: test_feed(sess, net)
            # if opts.resample_n is not None: test_sampling(sess, net)
            # else: test_core(sess, net)

    logging.info('#-X---------------------------------------------#')
    logging.info('#                Finish Testing                 #')
    logging.info('#-----------------------------------------------#')


# ######################################################################################################################
# ######################################################################################################################
# DEBUG
# --------
def __________________________DEBUG_____________________________(): pass # dummy function for PyCharm IDE

opts.batch_size = 5

# core code for training
def DEBUG_core(sess):
    logging.info('#-----------------------------------------------#')
    logging.info('#                Start Debugging                #')
    logging.info('#-----------------------------------------------#')

    batch_img, batch_label, batch_weights = data_layers.data_HDF5(args.dataset,
                                                                  opts.shape_img, opts.shape_label, opts.shape_weights,
                                                                  shuffle=False, batch_size=opts.batch_size,
                                                                  prefetch_threads=12, prefetch_n=40,
                                                                  resample_n=40,
                                                                  augment=True)

    for bb in range(opts.n_samples):
        r_batch_img = []
        for b in range(8):
            start = timer()
            e_batch_img, e_batch_label, e_batch_weights = sess.run([batch_img, batch_label, batch_weights])
            end = timer()

            print("got batch in %.4f s : img %s %s" % ((end - start), str(e_batch_img.shape), str(e_batch_img.dtype)))

            r_batch_img.append(np.reshape(e_batch_img, [-1, e_batch_img.shape[2], e_batch_img.shape[3]]))

        print("stitching and creating file")
        out_img = np.concatenate( [img_util.to_rgb(batch) for batch in r_batch_img] , axis=1)
        img_util.save_image(out_img, "%s/img_aug_%s.jpg" % (args.predict_dir, str(bb)))


    batch_img, batch_label, batch_weights = data_layers.data_HDF5(args.dataset,
                                                                  opts.shape_img, opts.shape_label, opts.shape_weights,
                                                                  shuffle=True, batch_size=opts.batch_size,
                                                                  prefetch_threads=12, prefetch_n=10,
                                                                  resample_n=None,
                                                                  augment=True)

    for b in range(50):
        start = timer()
        e_batch_img, e_batch_label, e_batch_weights = sess.run([batch_img, batch_label, batch_weights])
        end = timer()

        print("got batch in %.4f s : img %s %s, label %s %s, weights %s %s" % ((end - start),
                                                                                 str(e_batch_img.shape),
                                                                                 str(e_batch_img.dtype),
                                                                                 str(e_batch_label.shape),
                                                                                 str(e_batch_label.dtype),
                                                                                 str(e_batch_weights.shape),
                                                                                 str(e_batch_weights.dtype)))

        r_batch_img = np.reshape(e_batch_img, [-1, e_batch_img.shape[2], e_batch_img.shape[3]])
        r_batch_label = np.reshape(e_batch_label, [-1, e_batch_label.shape[2], e_batch_label.shape[3]])
        r_batch_weights = np.reshape(e_batch_weights, [-1, e_batch_weights.shape[2], e_batch_weights.shape[3]])

        out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                  np.squeeze(img_util.to_rgb(r_batch_label)),
                                  np.squeeze(img_util.to_rgb(r_batch_weights, normalize = True))), axis = 1)

        img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, str(b)))


#TODO: Remove debugging code
if opts.debug and False: # temporarily disabled
    logging.info('####################################################################')
    logging.info('#                           DEBUGGING                              #')
    logging.info('####################################################################')


    with tf.Session(config=tf_config) as sess:
        if opts.data_layer_type == 'hdf5' or opts.data_layer_type == 'feed_dict' or opts.data_layer_type is None:
            DEBUG_core(sess)
        elif opts.data_layer_type == 'tfrecords':
            with_queues(sess, DEBUG_core)
        elif opts.data_layer_type == 'hdf5_dset' or opts.data_layer_type == 'hdf5_tables':
            with_queues_and_loaders(sess, DEBUG_core)

    logging.info('#-X---------------------------------------------#')
    logging.info('#               Finish Debugging                #')
    logging.info('#-----------------------------------------------#')

logging.info('\n\n ... done :)')
