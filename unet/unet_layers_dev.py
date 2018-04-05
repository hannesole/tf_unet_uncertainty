# UNET LAYERS / BLOCKS
# ====================
#
# The two basic components of the U-Net Architecture:
#   Down_Block: 2 convolutions with 2x2 max pooling
#   Up_Bock: Upconvolution, concat (copy) of corresponding down layer and convolutions.
#
# A dropout layer is added in each block
#
# Author: Hannes Horneber
# Date: 2018-03-18

import tensorflow as tf
import tensorflow.contrib as tc

normalizer_params_default = {'decay': 0.99,
                             'is_training': None,
                             'zero_debias_moving_mean': False
                             }

def down_block(input_layer, n_features, is_training, keep_prob, name=None, no_max_pool=False, norm_fn=None, **kwargs):
    """
    Down block consisting of two Convolutions and a max pool.
    For Up Block-Concatinations the layer is returned before max_pooling (2nd return arg).

    :param input_layer: input
    :param n_features: number of features that are created by the conv
    :param is_training: phase
    :param keep_prob: for dropout layers
    :param name: optional name of the layer
    :param no_max_pool: last down_block of a Unet has no max_pool
    :param norm_fn: optional normalizer function for conv2d
    :param kwargs: normalizer_params for norm_fn (otherwise defaults if norm_fn is set)
    :return: 1st: the max_pool out_put | 2nd: layer for concat during expansion
    """
    # set normalizer params and set normalizer_params['is_training'] to is_training if not already set
    normalizer_params = kwargs.get('normalizer_params', normalizer_params_default) if norm_fn is not None else None
    if normalizer_params is not None and 'is_training' in normalizer_params \
            and normalizer_params.get('is_training') is None:
        normalizer_params['is_training'] = is_training # for this is_training needs to be set, otherwise error

    # create layers for block
    with tf.variable_scope('UNet/down_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
        down0a = tc.layers.conv2d(input_layer, n_features, (3, 3),
                                  normalizer_fn=norm_fn, normalizer_params=normalizer_params)
        down0b = tc.layers.conv2d(down0a, n_features, (3, 3),
                                  normalizer_fn=norm_fn, normalizer_params=normalizer_params)

        # only add dropout if keep_prob is not 1.
        if keep_prob < 1: down0b = tc.layers.dropout(down0b, keep_prob=keep_prob)

        # last down_block has no max_pool
        if no_max_pool:
            return down0b, down0b
        else:
            down0c = tc.layers.max_pool2d(down0b, (2, 2), padding='same')
            return down0c, down0b   # 1st: the max_pool out_put | 2nd: layer for concat during expansion


def up_block(input_layer, concat_layer, n_features, is_training, keep_prob, name=None, norm_fn=None, **kwargs):
    """
    Up block consisting of a deconvolution and three convolutions.

    :param input_layer: input
    :param concat_layer: layer that is copied for the convolution (usually output of the corresponding down block)
    :param n_features: number of features that are created by the conv
    :param is_training: phase
    :param keep_prob: for dropout layers
    :param name: optional name of the layer
    :param no_max_pool: last down_block of a Unet has no max_pool
    :param norm_fn: optional normalizer function for conv2d
    :param kwargs: normalizer_params for norm_fn (otherwise defaults if norm_fn is set)
    :return: 1st: the max_pool out_put | 2nd: layer for concat during expansion
    """
    # set normalizer params and set normalizer_params['is_training'] to is_training if not already set
    normalizer_params = kwargs.get('normalizer_params', normalizer_params_default) if norm_fn is not None else None
    if normalizer_params is not None and 'is_training' in normalizer_params \
            and normalizer_params.get('is_training') is None:
        normalizer_params['is_training'] = is_training # for this is_training needs to be set, otherwise error

    # create layers for block
    with tf.variable_scope('UNet/up_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
        # deconvolution
        up0a = tc.layers.conv2d_transpose(input_layer, n_features, (3, 3), 2,
                                          normalizer_fn=norm_fn, normalizer_params=normalizer_params)
        # concat; crop isn't needed due to padding='SAME'
        up0b = tf.concat([up0a, concat_layer], axis=3)
        # convolution
        up0c = tc.layers.conv2d(up0b, n_features, (3, 3),
                                normalizer_fn=norm_fn, normalizer_params=normalizer_params)
        up0d = tc.layers.conv2d(up0c, n_features, (3, 3),
                                normalizer_fn=norm_fn, normalizer_params=normalizer_params)
        up0e = tc.layers.conv2d(up0d, n_features, (3, 3),
                                normalizer_fn=norm_fn, normalizer_params=normalizer_params)

        # only add dropout if keep_prob is not 1.
        if keep_prob < 1: up0e = tc.layers.dropout(up0e, keep_prob=keep_prob)
        return up0e


# ######################################################################################################################
# w/o BATCH NORMALIZATION and DROPOUT
# -----------------------------------

# def down_block(input_layer, n_features, is_training, keep_prob, name=None, no_max_pool=False):
#     with tf.variable_scope('UNet/down_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         down0a = tc.layers.conv2d(input_layer, n_features, (3, 3))
#         down0b = tc.layers.conv2d(down0a, n_features, (3, 3))
#         #down0b_do = tc.layers.dropout(down0b, keep_prob=keep_prob)
#         if no_max_pool:
#             return down0b, down0b
#         else:
#             down0c = tc.layers.max_pool2d(down0b, (2, 2), padding='same')
#             return down0c, down0b   # 1st: the max_pool out_put | 2nd: layer for concat during expansion
#
#
# def up_block(input_layer, concat_layer, n_features, is_training, keep_prob, name=None):
#     with tf.variable_scope('UNet/up_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         up0a = tc.layers.conv2d_transpose(input_layer, n_features, (3, 3), 2)
#         up0b = tf.concat([up0a, concat_layer], axis=3)
#         up0c = tc.layers.conv2d(up0b, n_features, (3, 3))
#         up0d = tc.layers.conv2d(up0c, n_features, (3, 3))
#         up0e = tc.layers.conv2d(up0d, n_features, (3, 3))
#         #up0e = tc.layers.dropout(up0e, keep_prob=keep_prob)
#         return up0e


# ######################################################################################################################
# with tf.nn.layers
# -----------------------------------

# def down_block(input_layer, n_features, is_training, keep_prob, name=None, no_max_pool=False):
#     with tf.variable_scope('UNet/down_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         # filter is
#         down0a = nn_conv2d(input_layer, n_features, (3, 3))
#         down0b = nn_conv2d(down0a, n_features, (3, 3))
#         if no_max_pool:
#             return down0b, down0b
#         else:
#             down0c = tf.nn.max_pool(down0b, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#             return down0c, down0b   # 1st: the max_pool out_put | 2nd: layer for concat during expansion
#
#
# def up_block(input_layer, concat_layer, n_features, is_training, keep_prob, name=None):
#     with tf.variable_scope('UNet/up_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         #up0a = tc.layers.conv2d_transpose(input_layer, n_features, (3, 3), 2)
#         up0a = nn_deconv2d(input_layer, n_features, (3, 3), 2)
#         up0b = tf.concat([up0a, concat_layer], axis=3)
#         up0c = nn_conv2d(up0b, n_features, (3, 3))
#         up0d = nn_conv2d(up0c, n_features, (3, 3))
#         #up0e = nn_conv2d(up0d, n_features, (3, 3))
#         return up0d
#
#
#
# def nn_conv2d(input_layer, n_features, kernel=(3, 3),
#               name=None, keep_prob=None, initializer=tf.contrib.layers.xavier_initializer()):
#     """ Constructs a conv2d similar to tensorflow.contrib.layers.conv2d() """
#
#     with tf.variable_scope('Conv2D_%s_%s' % (str(input_layer.shape[-1]) if name is None else '',
#                                              str(n_features) if name is None else name + '_' + str(n_features))):
#         # create variables
#         weights = tf.get_variable("weights", shape=[kernel[0], kernel[1], input_layer.shape[-1], n_features],
#                                  initializer=initializer)
#         bias = tf.get_variable("bias", shape=[n_features], initializer=initializer)
#
#         # convolution, bias, relu
#         conv = tf.nn.conv2d(input_layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
#         conv_b = tf.nn.bias_add(conv, bias)
#         conv_br = tf.nn.relu(conv_b)
#
#         # add dropout if keep_prob is provided
#         if keep_prob is None:
#             return conv_br
#         else:
#             return tf.nn.dropout(conv_br, keep_prob=keep_prob)
#
#
# def nn_deconv2d(input_layer, n_features, kernel=(3, 3), stride=2,
#               name=None, keep_prob=None, initializer=tf.contrib.layers.xavier_initializer()):
#     """ Constructs a deconv2d similar to tensorflow.contrib.layers.conv2d_transpose() """
#
#     with tf.variable_scope('Deconv2D_%s_%s' % (str(input_layer.shape[-1]) if name is None else '',
#                                              str(n_features) if name is None else name + '_' + str(n_features))):
#
#         # create variables
#         weights = tf.get_variable("weights", shape=[kernel[0], kernel[1], n_features, input_layer.shape[-1]],
#                                  initializer=initializer)
#         bias = tf.get_variable("bias", shape=[n_features], initializer=initializer)
#         output_shape = input_layer.get_shape().as_list()
#         output_shape[1] = output_shape[1] * 2
#         output_shape[2] = output_shape[2] * 2
#         output_shape[3] = n_features
#
#            # [input_layer.shape[0], input_layer.shape[1]*stride, input_layer.shape[2]*stride, n_features]
#
#         # convolution, bias, relu
#         conv = tf.nn.conv2d_transpose(input_layer, filter=weights, output_shape=output_shape,
#                                       strides=[1, stride, stride, 1], padding='SAME')
#         conv_b = tf.nn.bias_add(conv, bias)
#         conv_br = tf.nn.relu(conv_b)
#
#         # add dropout if keep_prob is provided
#         if keep_prob is None:
#             return conv_br
#         else:
#             return tf.nn.dropout(conv_br, keep_prob=keep_prob)
