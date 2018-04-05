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

def down_block(input_layer, n_features, is_training, keep_prob, name=None, no_max_pool=False):
    with tf.variable_scope('UNet/down_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
        down0a = tc.layers.conv2d(input_layer, n_features, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                  normalizer_params={'is_training': is_training})
        down0b = tc.layers.conv2d(down0a, n_features, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                  normalizer_params={'is_training': is_training})
        down0b_do = tc.layers.dropout(down0b, keep_prob=keep_prob)
        if no_max_pool:
            return down0b_do, down0b
        else:
            down0c = tc.layers.max_pool2d(down0b_do, (2, 2), padding='same')
            return down0c, down0b   # 1st: the max_pool out_put | 2nd: layer for concat during expansion

def up_block(input_layer, concat_layer, n_features, is_training, keep_prob, name=None):
    with tf.variable_scope('UNet/up_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
        up0a = tc.layers.conv2d_transpose(input_layer, n_features, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
                                          normalizer_params={'is_training': is_training})
        up0b = tf.concat([up0a, concat_layer], axis=3)
        up0c = tc.layers.conv2d(up0b, n_features, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                normalizer_params={'is_training': is_training})
        up0d = tc.layers.conv2d(up0c, n_features, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                normalizer_params={'is_training': is_training})
        up0e = tc.layers.conv2d(up0d, n_features, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                normalizer_params={'is_training': is_training})
        up0e = tc.layers.dropout(up0e, keep_prob=keep_prob)
        return up0e


# ######################################################################################################################
# w/o BATCH NORMALIZATION
# -----------------------
#
# def down_block(input_layer, n_features, is_training, keep_prob, name=None, no_max_pool=False):
#     with tf.variable_scope('UNet/down_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         down0a = tc.layers.conv2d(input_layer, n_features, (3, 3))
#         down0b = tc.layers.conv2d(down0a, n_features, (3, 3))
#         down0b_do = tc.layers.dropout(down0b, keep_prob=keep_prob)
#         if no_max_pool:
#             return down0b_do, down0b
#         else:
#             down0c = tc.layers.max_pool2d(down0b_do, (2, 2), padding='same')
#             return down0c, down0b   # 1st: the max_pool out_put | 2nd: layer for concat during expansion
#
# def up_block(input_layer, concat_layer, n_features, is_training, keep_prob, name=None):
#     with tf.variable_scope('UNet/up_%s'%(str(n_features) if name is None else name + '_' + str(n_features))):
#         up0a = tc.layers.conv2d_transpose(input_layer, n_features, (3, 3), 2)
#         up0b = tf.concat([up0a, concat_layer], axis=3)
#         up0c = tc.layers.conv2d(up0b, n_features, (3, 3))
#         up0d = tc.layers.conv2d(up0c, n_features, (3, 3))
#         up0e = tc.layers.conv2d(up0d, n_features, (3, 3))
#         up0e = tc.layers.dropout(up0e, keep_prob=keep_prob)
#         return up0e