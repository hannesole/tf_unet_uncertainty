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
import logging
import numpy as np
from collections import deque
import os
import shutil

# ##################################################################################
# ##############                     UNET CLASS                         ############
# ##################################################################################

class UNet():
    def __init__(self,  # data specific settings
                 is_training=False, keep_prob=None,  # params for UNet architecture
                 n_contracting_blocks=5, n_start_features=8,
                 norm_fn=None, normalizer_params=None,
                 shape_img=[1024, 1024, 3], shape_label=None, shape_weights=None, n_class=2,  # architecture/data
                 resize=None, resize_method=None,
                 dataset_pth=None,  # data layer ...
                 data_layer_type='tfrecords',
                 batch_size=1, shuffle=False, augment=False,
                 prefetch_n=None, prefetch_threads=None,
                 resample_n=None,
                 copy_script_dir=None, debug=False  # other stuff
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
        if shape_label is None: shape_label = [shape_img[0], shape_img[1], 1]
        if shape_weights is None: shape_weights = [shape_img[0], shape_img[1], 1]

        # set keep_prob default during training, otherwise 1
        if keep_prob is None:
            self.keep_prob = 0.9 if is_training else 1.
        else:
            self.keep_prob = keep_prob  # Giving a param overrides this default
        self.is_training = is_training
        self.n_class = n_class

        self.resize = resize
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.init_objects = []  # data layer may generate objects to be initialized
        self.close_objects = []  # ... or closed when creating a session

        self.debug = debug

        # copy script to train_dir
        if copy_script_dir is not None:
            shutil.copy(__file__, copy_script_dir + os.sep + os.path.basename(__file__))
            # also copy affiliated sources (layers file)
            copy_layers_file = os.path.dirname(os.path.abspath(__file__)) + os.sep + "unet_layers_dev.py"
            shutil.copy(copy_layers_file, copy_script_dir + os.sep + "unet_layers_dev.py")
            logging.debug('Copied py source to: ' + copy_script_dir + os.sep + os.path.basename(__file__))

        logging.info('#-----------------------------------------------#')
        logging.info('#                 Creating Unet                 #')
        logging.info('#-----------------------------------------------#')

        ## DATA LAYER
        self.batch_img, self.batch_label, self.batch_weights = \
            self._build_data_layer(dataset_pth,  # data specific settings
                                   shape_img=shape_img, shape_label=shape_label, shape_weights=shape_weights,
                                   resize = resize, resize_method=resize_method,
                                   data_layer_type=data_layer_type,
                                   batch_size=batch_size, shuffle = shuffle, augment=augment,
                                   is_training=is_training,  # params for UNet architecture
                                   prefetch_n=prefetch_n, prefetch_threads=prefetch_threads,
                                   resample_n=resample_n
                                   )
        ## NETWORK
        self.output_mask, self.sigma, self.prediction \
            = self._build_net(self.batch_img, n_class=self.n_class,
                              n_features_start=n_start_features, n_blocks_down=n_contracting_blocks,
                              is_training=self.is_training, keep_prob=self.keep_prob,
                              norm_fn=norm_fn, normalizer_params=normalizer_params
                              )
        logging.info('Created UNet with Input size %s and Output Tensor %s' % (str(shape_img), str(self.output_mask)))

        ## OUTPUT TEST
        #if not is_training:
         #   self.output_mask = tf.nn.softmax(self.output_mask)

        # if any layer created init/close objects dependent on session, remind user to init/close them
        if not self.init_objects == [] or not self.close_objects == []:
            logging.info(
                '\n#################################################\n' +
                'Make sure to init objects when running net: %s\n' % (str(self.init_objects)) +
                'Make sure to close objects after running net: %s\n' % (str(self.close_objects)) +
                '#################################################'
            )




    # #########   TRAIN OPERATION    #########
    # ----------------------------------------
    def create_train_op(self, learning_rate, summary_imgs=False, optimizer='Adam'):
        '''
        Creates train op with loss, learning rate and optimization parameters.
        Includes creating summaries.

        :param learning_rate:
        :return:
        '''

        def resizer(image_batch, size=[256, 256]):
            return tf.image.resize_images(image_batch, size) if self.resize is None else image_batch

        with tf.variable_scope('train_op'):
            self.global_step = tf.train.get_or_create_global_step()

            if summary_imgs:  # takes a lot of space
                with tf.variable_scope('summary_imgs'):
                    tf.summary.image('0_img', resizer(self.batch_img[..., 0:3]))
                    tf.summary.image('0_label', resizer(tf.cast(self.batch_label, tf.float32)))
                    tf.summary.image('0_weights', resizer(self.batch_weights))
                    tf.summary.image('2_activations_out_c0', resizer(self.output_mask[..., 0, np.newaxis]))
                    tf.summary.image('2_activations_out_c1', resizer(self.output_mask[..., 1, np.newaxis]))

                    softmax = tf.nn.softmax(logits=self.output_mask)
                    tf.summary.image('3_softmax_c0', resizer(softmax[..., 0, np.newaxis]))
                    tf.summary.image('3_softmax_c1', resizer(softmax[..., 1, np.newaxis]))
                    max_softmax = tf.argmax(softmax, axis=-1)
                    max_softmax = tf.cast(max_softmax, dtype=tf.float32)
                    tf.summary.image('1_max_softmax', resizer(max_softmax[..., np.newaxis]))


            # LOSS FUNCTIONS
            # ------------------------
            with tf.variable_scope('losses'):
                self.batch_label_idx = tf.squeeze(self.batch_label, axis=-1)
                # store label one-hot encoded (one channel per class)
                # before tf.one_hot, tf.squeeze removes (empty) last dim, otherwise shape will be (batch_size, x, y, 1, n_classes):
                self.batch_label = tf.one_hot(tf.squeeze(self.batch_label, axis=-1), self.n_class, axis=-1)

                # if summary_imgs:
                #     with tf.variable_scope('summary_imgs'):
                #         tf.summary.image('label_one_hot_c0', tf.image.resize_images(self.batch_label[..., 0, np.newaxis], [256, 256]))
                #         tf.summary.image('label_one_hot_c1', tf.image.resize_images(self.batch_label[..., 1, np.newaxis], [256, 256]))

                with tf.variable_scope('softmax_cross_entropy_with_logits'):
                    # softmaxCE LOSS (cross entropy loss with softmax)
                    loss_softmaxCE = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_label,
                                                                             logits=self.output_mask)
                    # above returns shape [batch_size, img.shape[1], img.shape[2]]
                    # needs to be compatible with [batch_size, img.shape[1], img.shape[2] 1] for self.batch_weights
                    loss_softmaxCE = tf.expand_dims(loss_softmaxCE, axis=-1)

                    # weighted loss
                    loss_softmaxCE_w = tf.multiply(loss_softmaxCE, self.batch_weights)

                    if summary_imgs:
                        with tf.variable_scope('summary_imgs'):
                            tf.summary.image('softmax_CE', resizer(loss_softmaxCE))
                            tf.summary.image('softmax_CE_weighted', resizer(loss_softmaxCE_w))

                    loss_softmaxCE_w = tf.reduce_mean(loss_softmaxCE_w)
                    tf.summary.scalar('loss/weighted_softmax_cross_entropy', loss_softmaxCE_w)
                    loss_softmaxCE = tf.reduce_mean(loss_softmaxCE)
                    tf.summary.scalar('loss/softmax_cross_entropy', loss_softmaxCE)


                # set loss used for optimization
                self.loss = loss_softmaxCE_w
                tf.summary.scalar('loss/optimize_loss', self.loss)
            # ------------------------

            # # UNCERTAINTY
            # # --------------
            # with tf.variable_scope('uncertainty'):
            #
            #     #label = self.batch_label_idx
            #     label_one_hot = tf.cast(self.batch_label, tf.bool)# [batch_size, x, y, classes]
            #     logits = self.output_mask  # [batch_size, x, y, classes]
            #     sigma = self.sigma  # tf.ones(shape=tf.shape(logits), dtype=logits.dtype) # [1, x, y, classes]
            #     #sigma = tf.RandomNormal(shape=tf.shape(logits), dtype=logits.dtype)
            #
            #     def corrupt_logits(logits, sigma):
            #         gaussian = tf.random_normal(tf.shape(logits), mean=0.0, stddev=1.0, dtype=logits.dtype)
            #         noise = tf.multiply(gaussian, sigma) #
            #         return tf.add(logits, noise)
            #
            #
            #
            #     # logits_true_class = tf.gather_nd(logits, tf.cast(label, dtype=tf.int32))
            #     logits_true_class = tf.boolean_mask(logits, label_one_hot)
            #
            #     #def loss_alleatoric(logits, sigma, label_one_hot):
            #     T = 20
            #     loss_al = 0
            #     for t in range(T):
            #         sample_corrupt_logits = corrupt_logits(logits, sigma)
            #
            #         loss_al_inner = 0
            #         for c in range(logits.shape[-1]):
            #             loss_al_inner = loss_al + tf.exp(sample_corrupt_logits)
            #
            #         loss_al = loss_al - tf.boolean_mask(sample_corrupt_logits, label_one_hot) + tf.log(loss_al_inner)
            #
            # if summary_imgs:  # takes a lot of space
            #     with tf.variable_scope('summary_imgs'):
            #         tf.summary.image('sigma_c0', tf.image.resize_images(
            #             self.sigma[..., 0, np.newaxis], [256, 256]))
            #         tf.summary.image('sigma_c0', tf.image.resize_images(
            #             self.sigma[..., 1, np.newaxis], [256, 256]))
            #         tf.summary.image('corrupt_prediction_c0', tf.image.resize_images(
            #             sample_corrupt_logits[..., 0, np.newaxis], [256, 256]))
            #         tf.summary.image('corrupt_prediction_c0', tf.image.resize_images(
            #             sample_corrupt_logits[..., 1, np.newaxis], [256, 256]))

                #          logits = tf.constant([[.1, .2, .7],
                #                                [.3, .4, .3]])
                #          label_one_hot = tf.constant([[0, 0, 1],
                #                                       [0, 1, 0]])
                #          label_one_hot = tf.constant([[0, 0, 1],
                #                                       [0, 1, 0]])
                #          C = tf.constant([3, 1])
                #
                #          V = logits[b, x, y, c]
                #
                # [[['b0x0y0c0', 'b0x1y0c0']   ['b0x0y0c1' 'b0x1y0c1'
                #  ['b0x0y1c0' 'b0x1y1c0']]   'b0x0y1c1' 'b0x1y1c1']
                #
                #  ['b1x0y0c0' 'b1x1y0c0'   ['b1x0y0c1' 'b1x1y0c1'
                #  'b1x0y1c0' 'b1x1y1c0']   'b1x0y1c1' 'b1x1y1c1']]
                #
                #          labels[b, x, y, c]

            # LEARNING RATE
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                                global_step=self.global_step,
                                                                decay_rate=0.5,
                                                                decay_steps=50000)
                tf.summary.scalar('learning_rate', self.learning_rate)

            # OPTIMIZATION (backpropagation alg)
            self.train_op = tc.layers.optimize_loss(loss=self.loss,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=optimizer)

            # SUMMARY
            self.merged_summary = tf.summary.merge_all()
            return self.train_op, self.global_step, self.loss, self.merged_summary



    # #########   TEST OPERATION     #########
    # ----------------------------------------
    def create_test_op(self, session, summary_imgs=False):
        '''
        Creates test op with loss and summaries.

        :param learning_rate:
        :return:
        '''
        with tf.variable_scope('test_op'):
            self.global_step = tf.train.get_or_create_global_step()

            if summary_imgs:  # takes a lot of space on disk when wrting images
                with tf.variable_scope('summary_imgs'):
                    tf.summary.image('img', tf.image.resize_images(self.batch_img[..., 0:3], [256, 256]))
                    tf.summary.image('label',
                                     tf.image.resize_images(tf.cast(self.batch_label, tf.float32), [256, 256]))
                    tf.summary.image('weights', tf.image.resize_images(self.batch_weights, [256, 256]))
                    tf.summary.image('prediction_c0', tf.image.resize_images(
                        self.output_mask[..., 0, np.newaxis], [256, 256]))
                    tf.summary.image('prediction_c1', tf.image.resize_images(
                        self.output_mask[..., 1, np.newaxis], [256, 256]))

            # LOSS FUNCTIONS
            # ------------------------
            with tf.variable_scope('losses'):
                self.batch_label_idx = tf.squeeze(self.batch_label, axis=-1)
                # store label one-hot encoded (one channel per class)
                # before tf.one_hot, tf.squeeze removes (empty) last dim, otherwise shape will be (batch_size, x, y, 1, n_classes):
                self.batch_label = tf.one_hot(tf.squeeze(self.batch_label, axis=-1), self.n_class, axis=-1)

                # if summary_imgs:
                #     with tf.variable_scope('summary_imgs'):
                #         tf.summary.image('label_one_hot_c0', tf.image.resize_images(self.batch_label[..., 0, np.newaxis], [256, 256]))
                #         tf.summary.image('label_one_hot_c1', tf.image.resize_images(self.batch_label[..., 1, np.newaxis], [256, 256]))

                with tf.variable_scope('softmax_cross_entropy_with_logits'):
                    # softmaxCE LOSS (cross entropy loss with softmax)
                    loss_softmaxCE = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_label,
                                                                             logits=self.output_mask)
                    # above returns shape [batch_size, img.shape[1], img.shape[2]]
                    # needs to be compatible with [batch_size, img.shape[1], img.shape[2] 1] for self.batch_weights
                    loss_softmaxCE = tf.expand_dims(loss_softmaxCE, axis=-1)

                    # weighted loss
                    loss_softmaxCE_w = tf.multiply(loss_softmaxCE, self.batch_weights)

                    if summary_imgs:
                        with tf.variable_scope('summary_imgs'):
                            softmax = tf.nn.softmax(logits=self.output_mask)
                            tf.summary.image('softmax_c0',
                                             tf.image.resize_images(softmax[..., 0, np.newaxis], [256, 256]))
                            tf.summary.image('softmax_c1',
                                             tf.image.resize_images(softmax[..., 1, np.newaxis], [256, 256]))
                            max_softmax = tf.argmax(softmax, axis=-1)
                            tf.summary.image('max_softmax',
                                             tf.image.resize_images(max_softmax[..., np.newaxis], [256, 256]))
                            tf.summary.image('softmax_CE', tf.image.resize_images(loss_softmaxCE, [256, 256]))
                            tf.summary.image('softmax_CE_weighted',
                                             tf.image.resize_images(loss_softmaxCE_w, [256, 256]))

                    loss_softmaxCE_w = tf.reduce_mean(loss_softmaxCE_w)
                    tf.summary.scalar('loss/weighted_softmax_cross_entropy', loss_softmaxCE_w)
                    loss_softmaxCE = tf.reduce_mean(loss_softmaxCE)
                    tf.summary.scalar('loss/softmax_cross_entropy', loss_softmaxCE)

                # set loss used for optimization
                self.loss = loss_softmaxCE_w
                tf.summary.scalar('loss/optimize_loss', self.loss)
                # ------------------------

            # SUMMARY
            self.merged_summary = tf.summary.merge_all()
            return self.train_op, self.global_step, self.loss, self.merged_summary



    # #########   DATA LAYER    #########
    # -----------------------------------
    def _build_data_layer(self, dataset_pth=None,  # data specific settings
                          shape_img=None, shape_label=None, shape_weights=None,  # ...
                          resize=None, resize_method=None,
                          is_training=False,  # params for UNet architecture
                          data_layer_type='feed_dict',
                          batch_size=1, shuffle=False, augment=False,
                          prefetch_n=None, prefetch_threads=None, # ...
                          resample_n=None, valset_path=None  # other features
                          ):
        '''
        Builds a data layer and adds it to the graph. The type of the data layer is determined by data_layer_type.
        Available datalayers are tfrecords, hdf5_dset, hdf5_tables and feed_dict (default).
        Required Parameters depend on the data layer.

        Note:
        - some data layers require additional actions during session that cannot be wrapped by this function,
            since they require a tf.Session to be started. See datalayer doc for specifics.
        - The feed_dict layer cannot be (easily) used for training with tfutils.SimpleTrainer.

        :param dataset_pth: Path to the dataset. Required by all but feed_dict.
        :param shape_img: Shape of the img data. Required by tfrecords and feed_dict.
        :param shape_label: Shape of the label data. Will be inferred as [batch_size shape_img[0:2] 1] if not provided.
        :param shape_weights: Shape of the weight data. Will be inferred as [batch_size shape_img[0:2] 1] if not provided.
        :param batch_size: required by all, but is defaulted to 1 if not provided
        :param data_layer_type: defaults to feed_dict if not provided
        :param is_training: defaults to False (testing) if not provided
        :return:
        '''

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
                                          resample_n=resample_n)
            elif data_layer_type == 'hdf5_dset':
                #TODO: basically deprecated, use dataset API instead (hdf5)
                hdf5loader, hdf5reader, self.batch_img, self.batch_label, self.batch_weights = \
                    data_layers.data_HDF5Queue(dataset_pth, is_training=is_training,
                                               batch_size=batch_size, shuffle=shuffle)
                self.init_objects.append(hdf5loader)
                self.close_objects.append(hdf5reader)
            elif data_layer_type == 'hdf5_tables':
                # TODO: basically deprecated, use dataset API instead (hdf5)
                hdf5loader, hdf5reader, self.batch_img, self.batch_label, self.batch_weights = \
                    data_layers.data_HDF5TableQueue(dataset_pth, is_training=is_training,
                                                    batch_size=batch_size, shuffle=shuffle)
                self.init_objects.append(hdf5loader)
                self.close_objects.append(hdf5reader)
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
                   norm_fn=None, normalizer_params=None):
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

        ## OUTPUT LAYER
        with tf.variable_scope('UNet/output_mask'):
            # self.output_mask = fully connected layer at the end
            self.output_mask = tc.layers.conv2d(output_prev_layer, n_class, [1, 1], activation_fn=None)
            self.sigma = 0  # tc.layers.conv2d(output_prev_layer, n_class, [1, 1], activation_fn=None, name="sigma")

            if True: # not self.is_training:
                self.prediction = tf.argmax(tf.nn.softmax(self.output_mask), axis=-1, output_type=tf.int32)
                return self.output_mask, self.sigma, self.prediction
            else:
                return self.output_mask, self.sigma, None




    @property
    def vars(self):
        return [i for i in tf.global_variables_initializer() if 'UNet' in i.name]