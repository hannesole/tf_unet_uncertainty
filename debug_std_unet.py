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
import time
import matplotlib

matplotlib.use('Agg')
import numpy as np
import argparse
import shutil

from timeit import default_timer as timer

# check how long tensorflow import takes
print('Importing tensorflow ...')
t_start = timer()
import tensorflow as tf
print('Elapsed time: %.4f s' % (timer() - t_start))

import logging

from util import img_util
from util import logs
#from unet import model
from unet import model_dev as model
from unet import data_layers
from tfutils import SimpleTrainer

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
                    help='Provide a model checkpoint file, otherwise searching for last one in train_dir/checkpoints',
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
parser.add_argument('--predict_dir', '-p', metavar='predict_dir', required=False,
                    help='Prediction is written to \"predict_dir/\". ' +
                         'If not specified using prediction subfolder in train_dir.',
                    default=None)

args = parser.parse_args()


# ######################################################################################################################
# SCRIPT ARGS / OVERRIDE ARGS
# ---------------------------
# Since I usually call this from an IDE, not all script arguments are implemented as commandline arguments.
# You can also override commandline defaults here.
class opts(object):
    # wrapper class for all options (similar to a C++ struct)
    # -> General
    debug = True
    copy_script_dir = 'train_dir'

    # -> Training & Testing
    batch_size = 1  # simultaneous processing of batch_size images (GPU mem is limit)
    shuffle = True  # shuffle data input
    augment = True
    n_contracting_blocks = 5
    n_start_features = 32

    data_layer_type = 'hdf5'  # see unet model / data_layers for specifics
    shape_img = [1024, 1024, 4]  # of a single image
    shape_label = [1024, 1024, 1]  # of a single label mask
    shape_weights = [1024, 1024, 1]  # of a single weight mask

    # -> Training
    train = True  # script will train
    train_name = 'augment_saver'
    init_learning_rate = 0.001  # initial learning rate
    max_iter = 25000  # maximum iterations
    keep_prob = 0.8  # dropout - 1

    prefetch_n = batch_size * 20
    prefetch_threads = 12

    # -> Testing
    test = True  # script will do testing
    n_samples = 20  # ... of n_samples
    resample_n = None   # resample same image to calculate uncertainty, N or None
    keep_prob_test = keep_prob  # use 1.0 for no dropout during test-time
    pass

# -> Override CLI arguments (for convenience when running from IDE)
PROJECT_DIR = '/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/'

#args.output_dir = '/home/hornebeh/proj_tf_unet/output/'
args.name = 'unet_' + time.strftime("%Y-%m-%d_%H%M") + '_debug'
#args.name = 'overwrite'


#args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-03-25_1904_augment'
#args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-03-26_1835_alleatoric'
args.train_dir = PROJECT_DIR + 'output/' + 'unet_2018-03-27_1225_debug'

#args.train_dir = PROJECT_DIR + 'output_scr/' + 'unet_2018-03-25_2021_augment_saver'

#args.checkpoint = PROJECT_DIR + 'output/' + unet_2018-03-25_1904_augment/checkpoints/snapshot-4000"
#args.checkpoint = PROJECT_DIR + 'output_scr/' + unet_2018-03-22_2021_augment/checkpoints/snapshot-34000"

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
            shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    logging.info('For writing model and other output, created train_dir %s ' % train_dir)
else:
    # make sure train_dir exists
    train_dir = args.train_dir
    if not os.path.exists(train_dir):
        logging.info('For writing model and other output, created train_dir %s ' % train_dir)
        os.mkdir(train_dir)
    else:
        logging.info('For writing model and other output, using existing train_dir %s ' % train_dir)

# create predict_dir if it doesn't exist. Use either default naming or a given parameter.
if args.predict_dir is None:
    # create predict_dir with current time if none is specified
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
    # make sure predict_dir exists
    if not os.path.exists(args.predict_dir):
        os.mkdir(args.predict_dir)

# copy script to train_dir
if opts.copy_script_dir is not None:
    if opts.copy_script_dir == 'train_dir':
        opts.copy_script_dir = train_dir + os.sep + 'py_src'
    if not os.path.exists(opts.copy_script_dir):
        os.mkdir(opts.copy_script_dir)
    shutil.copy(__file__, opts.copy_script_dir + os.sep + os.path.basename(__file__))
    logging.debug('Copied py source to: ' + opts.copy_script_dir + os.sep + os.path.basename(__file__))


print(opts_to_str(opts))

# ######################################################################################################################
# TRAINING
# --------

# core code for training
def TRAIN_core(sess):
    pass

if opts.train:
    logging.info('####################################################################')
    logging.info('#                            TRAINING                              #')
    logging.info('####################################################################')

    # create network graph with data layer
    net = model.UNet(dataset_pth=args.dataset,
                     shape_img=opts.shape_img, shape_label=opts.shape_label, shape_weights=opts.shape_weights,
                     batch_size=opts.batch_size, shuffle=opts.shuffle, augment=opts.augment,
                     data_layer_type=opts.data_layer_type,
                     n_contracting_blocks=opts.n_contracting_blocks, n_start_features=opts.n_start_features,
                     is_training=True, keep_prob=opts.keep_prob,
                     prefetch_n=opts.prefetch_n, prefetch_threads=opts.prefetch_threads,
                     debug=opts.debug, copy_script_dir=opts.copy_script_dir
                     )

    from tensorflow.python import debug as tf_debug
    #with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=tf_config)) as sess:   # wrap tfdbg
    with tf.Session(config=tf_config) as sess:
        logging.info('#-----------------------------------------------#')
        logging.info('#               Starting Training               #')
        logging.info('#-----------------------------------------------#')
        # create a tfutils.SimpleTrainer that handles the mainloop and manages checkpoints
        trainer = SimpleTrainer(session=sess, train_dir=train_dir)
        # load model for continued training (if None provided, searches in train_dir, if not found doesn't load)
        trainer.load_checkpoint(args.checkpoint)

        # set train_op and summaries
        train_op, global_step, loss, merged_summary = net.create_train_op(0.001)

        # init variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # setup saver list
        saver_list = tf.trainable_variables() # might not contain all relevant variables -> tf.global_variables()
        saver_list.append(global_step)
        # import pprint; pprint.pprint(saver_list)

        trainer.mainloop(
            max_iter=400000,
            saver_interval=2000,
            saver_var_list=saver_list,
            train_ops=([train_op]),
            display_str_ops=[('Loss', loss)],
            display_interval=2,
            runstats_interval=100,
            trace_interval=100,
            summary_int_ops=[(1, merged_summary)]
        )

    logging.info('#-X---------------------------------------------#')
    logging.info('#                Finish Training                #')
    logging.info('#-----------------------------------------------#')


# ######################################################################################################################
# TESTING
# -------
if opts.train: tf.reset_default_graph()

def TEST_core(sess, net_test):
    pass

if opts.test:
    logging.info('####################################################################')
    logging.info('#                            TESTING                               #')
    logging.info('####################################################################')

    # create network graph with data layer
    net_test = model.UNet(dataset_pth=args.dataset,
                          shape_img=opts.shape_img, shape_label=opts.shape_label, shape_weights=opts.shape_weights,
                          batch_size=opts.batch_size, shuffle=False, augment=False,
                          data_layer_type=opts.data_layer_type,
                          n_contracting_blocks=opts.n_contracting_blocks, n_start_features=opts.n_start_features,
                          resample_n=opts.resample_n,
                          is_training=False, keep_prob=opts.keep_prob,
                          debug=opts.debug
                         )


    from tensorflow.python import debug as tf_debug
    #with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=tf_config)) as sess:  # wrap tfdbg
    with tf.Session(config=tf_config) as sess:
        logging.info('#-----------------------------------------------#')
        logging.info('#               Starting Testing                #')
        logging.info('#-----------------------------------------------#')

        trainer = SimpleTrainer(session=sess, train_dir=train_dir)
        # load model (if None provided, gets latest from train_dir/checkpoints, if none found doesn't load)
        #trainer.load_checkpoint(args.checkpoint)

        # init variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # ###########################################################################
        # RUN UNET
        # ###########################################################################
        logging.debug("predicting, sampling %s times, batch_size %s" % (opts.n_samples, opts.batch_size))

        for b in range(opts.n_samples):
            batch_img, batch_label, batch_prediction = sess.run(
                [net_test.batch_img, net_test.batch_label, net_test.output_mask])

            # out_img = np.squeeze(img_util.to_rgb(batch_prediction))
            # img_util.save_image(out_img, "%s/img_%s_pred.png" % (args.predict_dir, b))

            logging.debug('batch_prediction: %s %s' % (str(batch_prediction.shape), str(batch_prediction.dtype)))
            logging.debug('batch_img: %s %s' % (str(batch_img.shape), str(batch_img.dtype)))
            logging.debug('batch_label: %s %s' % (str(batch_label.shape), str(batch_label.dtype)))

            # logging.debug('describe prediction_samples: ' + str(stats.describe(batch_prediction)))
            # logging.debug('describe prediction_samples[0]: ' + str(stats.describe(prediction_samples[0])))
            # out_img = img_util.combine_img_prediction(batch_img, batch_label, batch_prediction)

            r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
            r_batch_label = np.reshape(batch_label, [-1, batch_label.shape[2], batch_label.shape[3]])
            r_batch_prediction = np.reshape(batch_prediction,
                                            [-1, batch_prediction.shape[2], batch_prediction.shape[3]])

            from sklearn.utils.extmath import softmax

            r_batch_softmax = np.zeros(r_batch_prediction.shape)
            r_batch_softmax[..., 0] = softmax(r_batch_prediction[..., 0])
            r_batch_softmax[..., 1] = softmax(r_batch_prediction[..., 1])
            argmax = np.argmax(r_batch_softmax, axis=-1)

            out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                      np.squeeze(img_util.to_rgb(r_batch_label)),
                                      np.squeeze(
                                          img_util.to_rgb(r_batch_prediction[..., 0, np.newaxis], normalize=True)),
                                      np.squeeze(
                                          img_util.to_rgb(r_batch_prediction[..., 1, np.newaxis], normalize=True)),
                                      np.squeeze(img_util.to_rgb(argmax[..., np.newaxis]))
                                      ), axis=1)

            img_util.save_image(out_img, "%s/img_%s.png" % (args.predict_dir, b))

            # ###########################################################################
            # CLOSE NET
            # ###########################################################################

    logging.info('#-X---------------------------------------------#')
    logging.info('#                Finish Testing                 #')
    logging.info('#-----------------------------------------------#')



# ######################################################################################################################
# ######################################################################################################################
# DEBUG
# --------
def ___________________________________________________():
    pass

# ######################################################################################################################
# SETUP FUNCTIONS FOR OTHER DATA LAYERS
# -------------------------------------

# wraps function fn with starting and stopping Coordinators and QueRunners
# sess is passed on to fn
def with_queues(sess, fn):
    # Some functions in the data layer add tf.train.QueueRunner objects to the graph.
    # To fill a queue, tf.train.start_queue_runners needs to be started.
    # Threads are coordinated by tf.train.Coordinator.
    # >> Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    fn(*sess)

    coord.request_stop()  # Stop the threads (for queues)
    coord.join(threads)  # Wait for threads (for queues) to stop


# wraps function fn with starting and stopping Coordinators and QueRunners
# as well as starting and stopping loaders and readers (as returned by the net data layer)
# sess is passed on to fn
def with_queues_and_loaders(sess, fn):
    loader = net.init_objects.pop()  # loader for HDF5 data queue
    with loader.begin(sess):  # needed for HDF5 data queue

        # >> Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        fn(*sess)

        coord.request_stop()  # Stop the threads (for queues)
        coord.join(threads)  # Wait for threads (for queues) to stop

    reader = net.close_objects.pop()  # reader from HDF5 data queue
    reader.close()  # needed for HDF5 data queue
    sess.close()



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

    for bb in range(19):
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
if opts.debug and False:
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
