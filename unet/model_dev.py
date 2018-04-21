# UNET MODEL
# ====================
#
# Unet Class that contains the data layer, the model network graph with its output tensors and a train op.
#
# NET: Represents the U-Net architecture. The model graph is build with a function _build_net().
# DATA_LAYER: Data layers are build separately with _build_data_layer(), based on parameters.
# TRAIN_OP: To train the model, a train_op needs to be created.
# Optimization parameters (loss, learning rate, optimizer type, ...) are defined there (and also summaries).
#
# Author: Hannes Horneber
# Date: 2018-03-18

import tensorflow as tf
import tensorflow.contrib as tc
from unet import data_layers
from unet import unet_layers_dev as unet_layers
from unet import uncertainty
from util import filesys
from util import img_util
from util.tfutils import helpers as tfutil_helpers
import logging
import numpy as np
from collections import deque
import os
import shutil
import time

# ##################################################################################
# ##############                     UNET CLASS                         ############
# ##################################################################################

class UNet():
    def __init__(self,  # data specific settings
                 is_training=False, keep_prob=None,  # params for UNet architecture
                 n_contracting_blocks=5, n_start_features=8,
                 norm_fn=None, normalizer_params=None,
                 aleatoric_sample_n=None,
                 copy_script_dir=None, debug=False,  # other stuff
                 opts_main=None, opts_val=None,  # for datalayer
                 train_dir=None,
                 sess=None  # for val_fn
                 ):
        '''
        Creates the Unet Model with TFRecords data layer. Most params are optional and have default values.
        However, to construct the datalayer at least, either shape_img is needed (for a 'feed_dict' data layer;
        shape_label and _weights are inferred if not provided)) or dataset_pth (for 'tfrecords' or 'hdf5' data layers).
        See _build_data_layer for specifics.

        :param dataset_pth: (needed for 'tfrecords' or 'hdf5' data layers) path to data
        :param shape_img: (needed for 'feed_dict' and 'tfrecords' data layer, for hdf5 layers inferred from data).
        How input data (img) is shaped, by default shape_img=[1024, 1024, 3]
        :param shape_label: (optional) how label is shaped, by default inferred from shape_img as [1024, 1024, 1]
        :param shape_weights: (optional) how weights are shaped, by default inferred from shape_img as [1024, 1024, 1]
        :param batch_size: (optional) by default 1
        :param data_layer_type: (optional) by default 'feed_dict' (shape_img needs to be provided)
        :param is_training: (optional) by default False
        :param keep_prob: (optional) 1 - dropout, by default 0.9 during training, otherwise 1. (means no dropout)
        :param n_contracting_blocks: (optional) by default 5, number of blocks of contracting path, see _build_net()
        :param n_start_features: (optional) by default 64, number of features in first block, see _build_net()
        :param copy_script_dir: (optional) if provided, the model.py file is copied to this directory.
        :param debug: (optional) False by default. If True, image.summaries (large!) are added to the event log.
        '''
        # set keep_prob default during training, otherwise 1
        if keep_prob is None:
            self.keep_prob = 0.9 if is_training else 1.
        else:
            self.keep_prob = keep_prob  # Giving a param overrides this default

        self.is_training = is_training
        # needed to decide whether to create a sigma output layer or not. Rest of options go to create_train_op.
        self.aleatoric_sample_n = aleatoric_sample_n

        # in case any op writes file output (e.g. val summary)
        self.train_dir = train_dir
        self.sess = sess

        # contains specific opts, e.g. for datalayers
        self.opts_main = opts_main  # opts.train or opts.test
        self.opts_val = opts_val    # opts only for validation

        # for datalayers/train_op
        self.n_class = opts_main.n_class
        self.resize = opts_main.resize

        self.debug = debug

        # copy script to train_dir
        if copy_script_dir is not None:
            copied = filesys.copy_file(__file__, copy_script_dir + os.sep + os.path.basename(__file__))
            logging.info('Copied model py source to: ' + copied)
            # also copy affiliated sources (layers file)
            copy_layers_file = os.path.dirname(os.path.abspath(__file__)) + os.sep + "unet_layers_dev.py"
            copied = filesys.copy_file(copy_layers_file, copy_script_dir + os.sep + os.path.basename(copy_layers_file))
            logging.info('Copied layers py source to: ' + copied)

        logging.info('#-----------------------------------------------#')
        logging.info('#                 Creating Unet                 #')
        logging.info('#-----------------------------------------------#')

        ## DATA LAYER
        def f_data_train():
            """ function to create datalayer for training """
            batch_img, batch_label, batch_weights = \
                self._build_data_layer(opts_main.dataset,
                                       shape_img=opts_main.shape_img, shape_label=opts_main.shape_label,
                                       shape_weights=opts_main.shape_weights,
                                       resize=opts_main.resize, resize_method=opts_main.resize_method,
                                       data_layer_type=opts_main.data_layer_type,
                                       batch_size=opts_main.batch_size,
                                       shuffle=opts_main.shuffle, augment=opts_main.augment,
                                       prefetch_n=opts_main.prefetch_n, prefetch_threads=opts_main.prefetch_threads,
                                       resample_n=opts_main.resample_n,
                                       is_training=is_training,
                                       name = 'train' if is_training else 'test'
                                       )
            return (batch_img, batch_label, batch_weights)

        def f_data_val():
            """ function to create datalayer for validation """
            batch_img, batch_label, batch_weights = \
                self._build_data_layer(opts_val.dataset,
                                       shape_img=opts_val.shape_img, shape_label=opts_val.shape_label,
                                       shape_weights=opts_val.shape_weights,
                                       resize=opts_val.resize, resize_method=opts_val.resize_method,
                                       data_layer_type=opts_val.data_layer_type,
                                       batch_size=opts_val.batch_size,
                                       shuffle=opts_val.shuffle, augment=opts_val.augment,
                                       prefetch_n=opts_val.prefetch_n, prefetch_threads=opts_val.prefetch_threads,
                                       resample_n=opts_val.resample_n,
                                       is_training=is_training,
                                       name = 'val'
                                       )
            return (batch_img, batch_label, batch_weights)

        if opts_val is None:
            # if validation options are not defined, only use training data_layer
            self.batch_img, self.batch_label, self.batch_weights = f_data_train()
        else:
            # if validation options are defined, create two data layers and feedable switch 'use_train_data'
            self.use_train_data = tf.placeholder_with_default(tf.constant(True, dtype=bool), [], name="is_training")
            self.batch_img, self.batch_label, self.batch_weights = \
                tf.cond(self.use_train_data, f_data_train, f_data_val, name='DataLayers')


        ## NETWORK
        self.output_mask, self.prediction, self.sigma_activations \
            = self._build_net(self.batch_img, n_class=self.n_class,
                              n_features_start=n_start_features, n_blocks_down=n_contracting_blocks,
                              is_training=self.is_training, keep_prob=self.keep_prob,
                              aleatoric_samples = aleatoric_sample_n,
                              norm_fn=norm_fn, normalizer_params=normalizer_params
                              )

        logging.info('Created UNet with input %s and output tensors %s%s'
                     % (str(self.batch_img.shape), str(self.output_mask.shape),
                        ('' if self.sigma_activations is None else (' | sigma %s'  % str(self.sigma_activations.shape)))))



    # #########   TRAIN OPERATION    #########
    # ----------------------------------------
    def create_train_op(self, opts, img_summary=True, metrics=True):
        '''
        Creates train op with loss, learning rate and optimization parameters.
        Includes creating summaries.

        :param opts: option object that
        allows access to options via opts.learning_rate_init, opts.optimizer ...
        should contain train options
        :return:
        '''
        with tf.variable_scope('Train_op'):
            self.global_step = tf.train.get_or_create_global_step()

            # LOSS FUNCTIONS
            # ------------------------
            # compute multiple losses but only use one for optimization
            with tf.variable_scope('losses'):
                # store label one-hot encoded (one channel per class)
                #  before tf.one_hot, tf.squeeze removes (empty) last dim, otherwise shape will be (batch_size, x, y, 1, n_classes):
                self.batch_label_one_hot = tf.one_hot(tf.squeeze(self.batch_label, axis=-1), self.n_class, axis=-1)

                # softmaxCE LOSS (cross entropy loss with softmax)
                softmaxCE = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_label_one_hot,
                                                                         logits=self.output_mask)

                # above returns shape [batch_size, img.shape[1], img.shape[2]]
                # needs to be compatible with [batch_size, img.shape[1], img.shape[2] 1] for self.batch_weights
                softmaxCE = tf.expand_dims(softmaxCE, axis=-1)
                # weighted loss
                softmaxCE_w = tf.multiply(softmaxCE, self.batch_weights)

                loss_softmaxCE_w = tf.reduce_mean(softmaxCE_w)
                tf.summary.scalar('loss/weighted_softmax_cross_entropy', loss_softmaxCE_w)
                loss_softmaxCE = tf.reduce_mean(softmaxCE)
                tf.summary.scalar('loss/softmax_cross_entropy', loss_softmaxCE)


                # set loss used for optimization
                if self.aleatoric_sample_n is None:
                    self.loss = loss_softmaxCE_w
                else:
                # UNCERTAINTY LOSS
                # -----------------
                    loss_aletaoric, sigma, sampled_logits = \
                        uncertainty.aleatoric_loss(self.output_mask, self.batch_label_one_hot,
                                                   self.sigma_activations, self.aleatoric_sample_n,
                                                   regularization=opts.aleatoric_reg)

                    self.loss = loss_aletaoric
                    tf.summary.scalar('loss/aleatoric_loss', self.loss)
                # -----------------

            # IMAGE SUMMARY
            # ------------------------
            # adds images to the summary. This might make the events file huge and make loading/writing slow.
            if img_summary:
                def resizer(image_batch, size=[128, 128]):
                    return tf.image.resize_images(image_batch, size) if self.resize is None else image_batch

                with tf.variable_scope('img_summary'): # numbered to control order in tensorboard
                    tf.summary.image('0_img', resizer(self.batch_img[..., 0:3]))
                    tf.summary.image('0_label', resizer(tf.cast(self.batch_label, tf.float32)))
                    tf.summary.image('0_weights', resizer(self.batch_weights))
                    tf.summary.image('2_activations_out_c0', resizer(self.output_mask[..., 0, np.newaxis]))
                    tf.summary.image('2_activations_out_c1', resizer(self.output_mask[..., 1, np.newaxis]))

                    softmax = tf.nn.softmax(logits=self.output_mask)
                    tf.summary.image('3_softmax_c0', resizer(softmax[..., 0, np.newaxis]))
                    tf.summary.image('3_softmax_c1', resizer(softmax[..., 1, np.newaxis]))
                    max_softmax = tf.argmax(softmax, axis=-1) # = self.prediction
                    max_softmax = tf.cast(max_softmax, dtype=tf.float32) # needed to be displayed by summary.image
                    tf.summary.image('1_segmentation', resizer(max_softmax[..., np.newaxis]))

                    if self.aleatoric_sample_n is not None:
                        averaged_sampled_logits = tf.reduce_mean(sampled_logits, axis=0)
                        tf.summary.image('2_corrupt_sm_logits_c0', resizer(averaged_sampled_logits[..., 0, np.newaxis]))
                        tf.summary.image('2_corrupt_sm_logits_c1', resizer(averaged_sampled_logits[..., 1, np.newaxis]))
                        tf.summary.image('1_sigma', resizer(sigma))
                        uncertainty_map = uncertainty.gaussian_entropy(sigma)
                        tf.summary.image('1_uncertainty_map', resizer(uncertainty_map))
                        # for uncertainty per class
                        # tf.summary.image('4_sigma_c0', resizer(sigma[..., 0, np.newaxis]))
                        # tf.summary.image('4_sigma_c1', resizer(sigma[..., 1, np.newaxis]))
                        # tf.summary.image('1_uncertainty_map_c0', resizer(uncertainty_map[..., 0, np.newaxis]))
                        # tf.summary.image('1_uncertainty_map_c1', resizer(uncertainty_map[..., 1, np.newaxis]))

                    tf.summary.image('4_softmax_CE', resizer(softmaxCE))
                    tf.summary.image('4_softmax_CE_weighted', resizer(softmaxCE_w))

            # LEARNING RATE / STATS
            # ------------------------
            with tf.variable_scope('stats'):
                # activation range
                activation_min = tf.reduce_min(self.output_mask)
                tf.summary.scalar('activation_min', activation_min)
                activation_max = tf.reduce_max(self.output_mask)
                tf.summary.scalar('activation_max', activation_max)
                activation_avg = tf.reduce_mean(self.output_mask)
                tf.summary.scalar('activation_avg', activation_avg)

                # sigma_activations range
                if self.aleatoric_sample_n is not None:
                    sigma_min = tf.reduce_min(self.sigma_activations)
                    tf.summary.scalar('sigma_min', sigma_min)
                    sigma_max = tf.reduce_max(self.sigma_activations)
                    tf.summary.scalar('sigma_max', sigma_max)
                    sigma_avg = tf.reduce_mean(self.sigma_activations)
                    tf.summary.scalar('sigma_avg', sigma_avg)

                if metrics:
                    #TODO all metrics are somehow always 0
                    # adjust prediction shape (is (1, ?, ?) vs label (1, ?, ?, 1))
                    prediction_4D = tf.expand_dims(self.prediction, axis=-1)

                    accuracy, _ = tf.metrics.accuracy(self.batch_label, prediction_4D)
                    tf.summary.scalar('z_accuracy', accuracy)
                    mean_iou, _ = tf.metrics.mean_iou(self.batch_label, prediction_4D, self.n_class)
                    tf.summary.scalar('z_mean_iou', mean_iou)
                    precision, _ = tf.metrics.precision(self.batch_label, prediction_4D)
                    tf.summary.scalar('z_precision', precision)
                    recall, _ = tf.metrics.recall(self.batch_label, prediction_4D)
                    tf.summary.scalar('z_recall', recall)

                # LEARNING RATE
                self.learning_rate = tf.train.exponential_decay(opts.learning_rate_init,
                                                                global_step=self.global_step,
                                                                decay_rate=opts.learning_rate_decay,
                                                                decay_steps=opts.max_iter)
                tf.summary.scalar('Learning_rate', self.learning_rate)
            # ------------------------

            # OPTIMIZATION (backpropagation alg)
            self.train_op = tc.layers.optimize_loss(loss=self.loss,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=opts.optimizer)

            # merge summaries to have only one op to rule them all
            self.merged_summary = tf.summary.merge_all()
            return self.train_op, self.global_step, self.loss, self.merged_summary


    # #########     VALIDATION       #########
    # ----------------------------------------
    def setup_val_ops(self):
        """ Needs to be run if net is supposed to do validation during training. """
        if self.opts_val is None:
            return []
        else:
            # setup a summary writer if one or both val ops are req
            if (self.opts_val.val_intervall is not None and self.opts_val.val_intervall > 0) or \
                    ((self.opts_val.val_intervall_sample is not None and self.opts_val.val_intervall_sample > 0) and\
                    (self.opts_val.n_samples is not None and self.opts_val.n_samples > 0)):
                train_logdir = os.path.join(self.train_dir, 'trainlogs/val')
                self.val_summary_writer = tf.summary.FileWriter(train_logdir, graph=tf.get_default_graph())

            val_ops = []
            # basic val_op (one sample, merged training summary)
            if self.opts_val.val_intervall is not None and self.opts_val.val_intervall > 0:
                val_ops.append( (self.opts_val.val_intervall, self.val_minimal_fn) )
            # val_sample_fn allows to sample multiple times, write out images. add custom code here best.
            if (self.opts_val.val_intervall_sample is not None and self.opts_val.val_intervall_sample > 0) and\
                    (self.opts_val.n_samples is not None and self.opts_val.n_samples > 0):
                val_ops.append( (self.opts_val.val_intervall_sample, self.val_sample_fn) )
                # create
                if self.opts_val.val_dir is not None:
                    self.val_dir = filesys.find_or_create_val_dir(train_dir=self.train_dir,
                                                                  val_dir=self.opts_val.val_dir)
            return val_ops


    def val_minimal_fn(self):
        """ does only one single forward pass """
        sess = self.sess
        global_step_value = self.sess.run(self.global_step)

        # only add summary for loss (run session with feed_dict to switch datalayer)
        val_summary,_,_,_ = sess.run([self.merged_summary, self.batch_img, self.batch_label, self.output_mask],
                               feed_dict={self.use_train_data: False} )

        self.val_summary_writer.add_summary(val_summary, global_step_value)
        self.val_summary_writer.flush()
        pass


    def val_sample_fn(self):
        """ does opts_val.n_samples forward passes and writes images """
        opts_val = self.opts_val
        val_dir = self.val_dir
        sess = self.sess

        if opts_val is not None and opts_val.val_sample_intervall is not None:
            logging.info('#----------------------------#')
            logging.info('#    Sampling over Valset    #')
            logging.info("# %s samples, to %s" % (opts_val.n_samples, val_dir))

            global_step_value = self.sess.run(self.global_step)
            # not needed:
            # self.val_summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=global_step_value)

            for b in range(opts_val.n_samples):
                try:
                    # run session with feed_dict to switch datalayer
                    batch_img, batch_label, batch_activations, batch_prediction, \
                    val_summary = \
                        sess.run(
                        [self.batch_img, self.batch_label, self.output_mask, self.prediction,
                         self.merged_summary],
                        feed_dict = {self.use_train_data : False}
                    )
                except tf.errors.OutOfRangeError:
                    break

                # add summaries
                self.val_summary_writer.add_summary(val_summary, global_step_value)
                self.val_summary_writer.flush()

                if val_dir is not None:
                    r_batch_img = np.reshape(batch_img, [-1, batch_img.shape[2], batch_img.shape[3]])
                    r_batch_label = np.reshape(batch_label, [-1, batch_label.shape[2], batch_label.shape[3]])
                    r_batch_activations = np.reshape(batch_activations,
                                                     [-1, batch_activations.shape[2], batch_activations.shape[3]])
                    r_batch_prediction = np.reshape(batch_prediction, [-1, batch_prediction.shape[2]])

                    out_img = np.concatenate((np.squeeze(img_util.to_rgb(r_batch_img)),
                                              np.squeeze(img_util.to_rgb(r_batch_label)),
                                              np.squeeze(
                                                  img_util.to_rgb(r_batch_activations[..., 0, np.newaxis], normalize=True)),
                                              np.squeeze(
                                                  img_util.to_rgb(r_batch_activations[..., 1, np.newaxis], normalize=True)),
                                              np.squeeze(img_util.to_rgb(r_batch_prediction[..., np.newaxis]))
                                              ), axis=1)

                    img_util.save_image(out_img, "%s/img_%s.png" % (val_dir, b))

            logging.info('#                            #')
            logging.info('#-X--------------------------#')
        pass


    # #########   TEST OPERATION     #########
    # ----------------------------------------
    def test_op(self):
        '''
        Creates test op that returns data, label and prediction batch tensors
        as well as certain metrics

        :param learning_rate:
        :return:
        '''

        self.global_step = tf.train.get_or_create_global_step()
        # prediction is (1, ?, ?), label (1, ?, ?, 1)
        prediction_4D = tf.expand_dims(self.prediction, axis=-1)

        accuracy = tf.metrics.accuracy(self.batch_label, prediction_4D)
        accuracy_per_class = tf.metrics.mean_per_class_accuracy(self.batch_label, prediction_4D, self.n_class)
        mean_iou = tf.metrics.mean_iou(self.batch_label, prediction_4D, self.n_class)
        precision = tf.metrics.precision(self.batch_label, prediction_4D)
        recall = tf.metrics.recall(self.batch_label, prediction_4D)

        softmax = tf.nn.softmax(logits=self.output_mask)

        return [self.batch_img, self.batch_label, self.batch_weights,
                self.output_mask, softmax, self.prediction,
                accuracy, precision, recall,
                accuracy_per_class, mean_iou
                ]



    # #########   DATA LAYER    #########
    # -----------------------------------
    def _build_data_layer(self, dataset_pth=None,  # data specific settings
                          shape_img=None, shape_label=None, shape_weights=None,  # ...
                          resize=None, resize_method=None,
                          is_training=False,  # params for UNet architecture
                          data_layer_type='feed_dict',
                          batch_size=1, shuffle=False, augment=False,
                          prefetch_n=None, prefetch_threads=None, # ...
                          resample_n=None, # other features
                          name = None
                          ):
        '''
        Builds a data layer and adds it to the graph. The type of the data layer is determined by data_layer_type.
        Available datalayers are tfrecords (deprecated, using queues, needs to be ported to tf.dataset),
        hdf5 and feed_dict (default).
        Required Parameters depend on the data layer.

        Note:
        - some data layers require additional actions during session that cannot be wrapped by this function,
            since they require a tf.Session to be started. See datalayer doc for specifics
            (deprecated, only applies to queues, i.e. tfrecords layer)
        - The feed_dict layer cannot be (easily) used for training with tfutils.SimpleTrainer.
            you may need to write your own main_loop or modify SimpleTrainer.
            feed_dict is not recommended for training anyways though (supposedly slow).s

        :param dataset_pth: Path to the dataset. Required by all but feed_dict.
        :param shape_img: Shape of the img data. Required by tfrecords and feed_dict.
        :param shape_label: Shape of the label data. Will be inferred as [batch_size shape_img[0:2] 1] if not provided.
        :param shape_weights: Shape of the weight data. Will be inferred as [batch_size shape_img[0:2] 1] if not provided.
        :param batch_size: required by all, but is defaulted to 1 if not provided
        :param data_layer_type: defaults to feed_dict if not provided
        :param is_training: defaults to False (testing) if not provided
        :return:
        '''
        if shape_label is None: shape_label = [shape_img[0], shape_img[1], 1]
        if shape_weights is None: shape_weights = [shape_img[0], shape_img[1], 1]

        with tf.variable_scope('DataLayer_' + data_layer_type + '/' + (
                'train_data' if is_training else 'test_data')):
            if data_layer_type == 'tfrecords':
                self.batch_img, self.batch_label, self.batch_weights = \
                    data_layers.data_TFRqueue(dataset_pth,
                                              shape_img, shape_label=shape_label, shape_weights=shape_weights,
                                              is_training=is_training, batch_size=batch_size, shuffle = shuffle)
            elif data_layer_type == 'hdf5':
                self.batch_img, self.batch_label, self.batch_weights = \
                    data_layers.data_HDF5(dataset_pth,
                                          shape_img=shape_img, shape_label=shape_label, shape_weights=shape_weights,
                                          is_training=is_training,
                                          batch_size=batch_size, shuffle=shuffle, augment=augment,
                                          prefetch_n=prefetch_n, prefetch_threads=prefetch_threads,
                                          resample_n=resample_n, name=name)
            else:  # data_layer_type=='feed_dict':
                self.batch_img, self.batch_label, self.batch_weights = \
                    data_layers.data_tf_placeholder(
                                        shape_img=shape_img, shape_label=shape_label, shape_weights=shape_weights,
                                        is_training=is_training, batch_size=batch_size)

            if resize is not None:
                if resize_method == "scale":
                    self.batch_img = tf.image.resize_images(self.batch_img, resize)
                    self.batch_label = tf.image.resize_images(self.batch_label, resize)
                    self.batch_label = tf.cast(self.batch_label, tf.uint8) # is changed by resize
                    self.batch_weights = tf.image.resize_images(self.batch_weights, resize)
                elif resize_method == "center_crop":
                    self.batch_img = tf.map_fn(lambda img: tf.image.central_crop(img, 0.5),
                                              self.batch_img, parallel_iterations=8, name="center_crop")
                    self.batch_label = tf.map_fn(lambda img: tf.image.central_crop(img, 0.5),
                                              self.batch_label, parallel_iterations=8, name="center_crop")
                    self.batch_weights = tf.map_fn(lambda img: tf.image.central_crop(img, 0.5),
                                              self.batch_weights, parallel_iterations=8, name="center_crop")
                else:
                    random_seed = 42 # tf.random_uniform(1, minval=0, maxval=65536, dtype=tf.int16)
                    self.batch_img = tf.random_crop(self.batch_img,
                                       [self.batch_img.get_shape().as_list()[0], resize[0], resize[1],
                                        self.batch_img.get_shape().as_list()[3]], seed=random_seed)
                    self.batch_label = tf.random_crop(self.batch_label,
                                       [self.batch_label.get_shape().as_list()[0], resize[0], resize[1],
                                        self.batch_label.get_shape().as_list()[3]], seed=random_seed)
                    self.batch_weights = tf.random_crop(self.batch_weights,
                                       [self.batch_weights.get_shape().as_list()[0], resize[0], resize[1],
                                        self.batch_weights.get_shape().as_list()[3]], seed=random_seed)

            return self.batch_img, self.batch_label, self.batch_weights


    # #########   NETWORK GRAPH    #########
    # --------------------------------------
    def _build_net(self, output_prev_layer, n_features_start, n_blocks_down, is_training, keep_prob, n_class,
                   aleatoric_samples=None, norm_fn=None, normalizer_params=None):
        '''
        Parametrized build of Unet model (flexible in depth and number of features)

        :param input: input to UNet block, usually the data layer
        :param n_features_start: how many features the first layer will have.
        The middle (deepest) layer will have n_features_start * 2^n_blocks_down features.
        (E.g. start with 8 features, with 4 blocks the deepest conv layer has 1024
        :param n_blocks_down: nr of blocks for contraction path
        n-1 blocks are added for expansion path, UNet has total of 2n - 1 blocks
        :return:
        '''
        concat_layers = deque()  # we need a LIFO stack as the FIRST downconv concat_layer is needed in the LAST upconv

        logging.info('Building UNet with %s blocks contracting path (starting features %s, input size %s x %s)'
                     % (str(n_blocks_down), str(n_features_start),
                        str(output_prev_layer.shape[1]), str(output_prev_layer.shape[2]) ))

        ## CONTRACTING PATH
        for i in range(0, n_blocks_down):
            n_features = n_features_start * (2 ** i)
            logging.info('AddLayer: UNet/down_%s with features %s' % (str(i), str(n_features)))
            output_prev_layer, concat_layer = unet_layers.down_block(output_prev_layer, n_features, is_training,
                                                                     keep_prob,
                                                                     norm_fn=norm_fn, normalizer_params=normalizer_params,
                                                                     name='fullsize' if i == 0 else str(i),
                                                                     no_max_pool=i == n_blocks_down-1) # for last block
            if not i == n_blocks_down - 1: concat_layers.append(concat_layer)  # for last down_block no concat is added

        ## EXPANDING PATH
        for i in range(n_blocks_down - 2, -1, -1):
            n_features = n_features_start * (2 ** i)
            logging.info('AddLayer: UNet/up_%s with features %s' % (str(i), str(n_features)))
            output_prev_layer = unet_layers.up_block(output_prev_layer, concat_layers.pop(), n_features, is_training,
                                                     keep_prob,
                                                     norm_fn=norm_fn, normalizer_params=normalizer_params,
                                                     name='fullsize' if i == 0 else str(n_blocks_down - 2 - i))

        ## OUTPUT LAYER(S)
        with tf.variable_scope('UNet/output_mask'):
            # self.output_mask = reduce feature space (map features to n_classes, essentially a classifier)
            output_mask = tc.layers.conv2d(output_prev_layer, n_class, [1, 1], activation_fn=None)
            # add sigma_activations for aleatoric loss (learned uncertainty map)
            if aleatoric_samples is not None:
                # per class sigma (per class uncertainty)
                #sigma_activations = tc.layers.conv2d(output_prev_layer, n_class, [1, 1], activation_fn=None)
                # per pixel uncertainty
                sigma_activations = tc.layers.conv2d(output_prev_layer, 1, [1, 1], activation_fn=None)
            else:
                sigma_activations = None

            # prediction is the predicted label mask/segmentation (essentialy argmax(softmax))
            prediction = tf.argmax(tf.nn.softmax(output_mask), axis=-1, output_type=tf.int32)
            return output_mask, prediction, sigma_activations


    @property
    def vars(self):
        return [i for i in tf.global_variables_initializer() if 'UNet' in i.name]