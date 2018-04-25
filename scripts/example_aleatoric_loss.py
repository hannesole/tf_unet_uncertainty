# ALEATORIC LOSS
# =========================
#
# Example and test code for calculating aleatoric loss.
#
# Author: Hannes Horneber
# Date: 2018-04-13

import tensorflow as tf
import logging
import os

tf_config = tf.ConfigProto(log_device_placement=False)
if 'dacky' in os.uname()[1]:
    logging.info('Dacky: Running with memory usage limits')
    # change tf_config for dacky to use only 1 GPU
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    # change tf_config for lmb_cluster so that GPU is visible and utilized
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.Session(config=tf_config)
tf.set_random_seed(1)

# how many aleatoric samples
T_samples = 2
# dim for n_classes
axis_c = 1

log0 = tf.log(0.0)
print('log(0) ' + str(log0.shape))
print(sess.run(log0))

##############################################################################
# SETUP
##############################################################################

sigma = tf.constant(
        [[[[0.040639675,0.08575996,0.05539584,0.08587229],
            [0.04121536,0.03663087,0.065721333,0.08779752],
            [0.08690897,0.04222715,0.035364604,0.023803174],
            [0.06833863,0.04921664,0.07013209,0.06710596,]],

            [[0.05988817,0.04168774,0.09096178,0.06016035,],
            [0.014121377,0.0634094,0.093972814,0.055255306],
            [0.019920528,0.054302454,0.067384136,0.045836174],
            [0.084448314,0.058103275,0.03344282,0.099018466]],

            [[0.048891675,0.05658355,0.086220324,0.030377162],
            [0.079628694,0.08301997,0.037294245,0.061025727],
            [0.03026377,0.08446696,0.08468499,0.01163864],
            [0.06305574,0.09808638,0.08773422,0.05724373,]]]])
# sigma_activations = tf.multiply(sigma_activations, 0)
print('sigma_activations ' + str(sigma.shape))
print(sess.run(sigma))
print('#' * 40)

label_one_hot = tf.constant(
    [[ [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1]],

       [[0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 0, 0]],

       [[1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0]] ]], dtype=tf.float32)
print('label one hot ' + str(label_one_hot.shape))
print(sess.run(label_one_hot))
print('#' * 40)

label_one_hot_neg = (label_one_hot - 1)*(-1)


def sample_corrupt_logits(logits, sigma, sample_n):
    new_shape = [sample_n, tf.shape(logits)[0],tf.shape(logits)[1], tf.shape(logits)[2], tf.shape(logits)[3]]
    gaussian = tf.random_normal(new_shape, mean=0.0, stddev=1.0, dtype=logits.dtype, seed=1)
    noise = tf.multiply(gaussian, sigma)  #
    return tf.add(logits, noise)

logits = tf.constant(
    [[[[0.05554414,0.01869845,0.07080972,0.27141213],
        [0.65317714,0.58058167,0.93273187,0.4293102,],
        [0.41135514,0.44323444,0.28234375,0.399562,],
        [0.9464109,0.7459034,0.5850476,0.9211211,]],

        [[0.68897665,0.3992982,0.28653347,0.22129631],
        [0.98678243,0.2548753,0.59044313,0.87323976],
        [0.35679007,0.40958285,0.13033354,0.46951437],
        [0.2529379,0.18824983,0.736686,0.48727047]],

        [[0.5001017,0.32503247,0.49028778,0.51371074],
        [0.20151675,0.78892636,0.653041,0.44452262],
        [0.13248003,0.3896109,0.902364,0.3857646,],
        [0.31667006,0.16516733,0.43019414,0.12097478]]]])
print('logits ' + str(logits.shape))
print(sess.run(logits))

print('#' * 60)

# ##############################################################################
# # SOFTMAX CROSS ENTROPY DEMYSTIFIED
# ##############################################################################
softmax = tf.nn.softmax(logits, dim=axis_c)
print('softmax ' + str(softmax.shape))
print(sess.run(softmax))
print('#' * 60)

# exps = tf.exp(logits - tf.reduce_max(logits, axis=axis_c))
# softmax_self = tf.divide(exps, tf.reduce_sum(exps, axis=axis_c))
# print('softmax_self ' + str(softmax_self.shape))
# print(sess.run(softmax_self))
# print('#' * 60)
#
logits_true_class = tf.multiply(softmax, label_one_hot)
# print('logits_true_class ' + str(logits_true_class.shape))
# print(sess.run(logits_true_class))
# print('#' * 60)

# reduce since all other entries are 0 (because of one_hot_label_mask)
logits_true_class_rs = tf.reduce_sum(logits_true_class, axis=axis_c, keep_dims=True)
# print('logits_true_class_rs ' + str(logits_true_class_rs.shape))
# print(sess.run(logits_true_class_rs))
# print('#' * 60)

softmax_cross_entropy_self = tf.multiply(tf.log(logits_true_class_rs), -1)
print('softmax_cross_entropy_self ' + str(softmax_cross_entropy_self.shape))
print(sess.run(softmax_cross_entropy_self))
print('#' * 60)
#
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=logits, dim=1)
print('softmax_cross_entropy ' + str(softmax_cross_entropy.shape))
print(sess.run(softmax_cross_entropy))
print('#' * 25)
loss = tf.reduce_mean(softmax_cross_entropy)
print('loss ' + str(loss.shape))
print(sess.run(loss))
print('#' * 25)

##############################################################################
# ADD NOISE

sampled_logits = sample_corrupt_logits(logits, sigma, T_samples)
print('sampled_corrupted_logits ' + str(sampled_logits.shape))
print(sess.run(sampled_logits))
# dim for n_classes changes when adding sample dim
axis_c = 2

# squash with softmax (optional)
# sampled_softmax_logits = tf.nn.softmax(sampled_logits, dim = axis_c)
# print('sampled_corrupted_softmax_logits ' + str(sampled_softmax_logits.shape))
# print(sess.run(sampled_softmax_logits))
# print('#' * 40)

print('#' * 60)
##############################################################################

# doesn't work 5-dimensional
# softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=label_one_hot, logits=sampled_logits, dim=axis_c)
# print('softmax_cross_entropy ' + str(softmax_cross_entropy.shape))
# print(sess.run(softmax_cross_entropy))
# print('#' * 25)

softmax_sample_logits = tf.nn.softmax(sampled_logits, dim=axis_c)
print('softmax_sample_logits ' + str(softmax_sample_logits.shape))
print(sess.run(softmax_sample_logits))
print('#' * 25)

##############################################################################
# CROSSENTROPY
#  select (softmaxed) logits (simulating a bool mask by setting unneeded values to zero)
#  then negative log

sm_logits_true_class = tf.multiply(softmax_sample_logits, label_one_hot)
# print('sm_logits_true_class ' + str(sm_logits_true_class.shape))
# print(sess.run(sm_logits_true_class))
# print('#' * 60)

# reduce since all other entries are 0 (because of one_hot_label_mask)
sm_logits_true_class_rs = tf.reduce_sum(sm_logits_true_class, axis=axis_c, keep_dims=True)
# print('sm_logits_true_class_rs ' + str(sm_logits_true_class_rs.shape))
# print(sess.run(sm_logits_true_class_rs))
# print('#' * 60)

sampled_sm_cross_entropy_self = tf.multiply(tf.log(sm_logits_true_class_rs), -1)
print('sampled_sm_cross_entropy_self ' + str(sampled_sm_cross_entropy_self.shape))
print(sess.run(sampled_sm_cross_entropy_self))
print('#' * 60)
loss1 = tf.reduce_mean(sampled_sm_cross_entropy_self)
print('loss1 ' + str(loss1.shape))
print(sess.run(loss1))
print('#' * 25)

# ##############################################################################
# # SUMMING
#
# sum1_exp = tf.exp(sampled_logits)  # was logits_other_classes
# print('sum1_exp ' + str(sum1_exp.shape))
# print(sess.run(sum1_exp))
#
# sum1_exp_rm1 = tf.exp(sampled_logits) - label_one_hot # after exp() tensor has 1 where was 0
# print('sum1_exp_rm0 ' + str(sum1_exp_rm1.shape))
# print(sess.run(sum1_exp_rm1))
#
# sum1_rs = tf.reduce_sum(sum1_exp_rm1, axis=axis_c, keep_dims=True)
# print('sum1_rs ' + str(sum1_rs.shape))
# print(sess.run(sum1_rs))
#
# sum1_log = tf.log(sum1_rs)
# print('sum1_log ' + str(sum1_log.shape))
# print(sess.run(sum1_rs))
#
# print('#' * 25)
# # sum1: over other classes
# sum1_other_classes = tf.log(tf.reduce_sum(tf.exp(sampled_logits) - label_one_hot, axis=axis_c, keep_dims=True))
# print('sum1_other_classes ' + str(logits_true_class_rs.shape))
# print(sess.run(sum1_other_classes))
# print('#' * 25)
#
#
# print('#' * 60)
#
# sum2_diff = logits_true_class_rs - sum1_other_classes
# print('sum2_diff ' + str(sum2_diff.shape))
# print(sess.run(sum2_diff))
#
# sum2_exp = tf.exp(sum2_diff)
# print('sum2_exp ' + str(sum2_exp.shape))
# print(sess.run(sum2_exp))
#
# print('#' * 25)
# # sum2: over samples. divide by number of samples -> reduce mean
# sum2_samples = tf.reduce_mean(tf.exp(logits_true_class_rs - sum1_other_classes), axis=0)
# print('sum2_samples ' + str(sum2_samples.shape))
# print(sess.run(sum2_samples))
# print('#' * 25)
#
# print('#' * 60)
# sum3_pixels = tf.log(sum2_samples)
# print('sum3_pixels ' + str(sum3_pixels.shape))
# print(sess.run(sum3_pixels))
#
# print('#' * 25)
# # sum over and divide by number of pixels
# loss_aletaoric = tf.reduce_mean(tf.log(sum2_samples))
# print('loss_aletaoric ' + str(loss_aletaoric.shape))
# print(sess.run(loss_aletaoric))
# print('#' * 25)


# loss_aletaoric = tf.reduce_sum(tf.log(sum2_samples)) --> inf
# loss_aletaoric = tf.reduce_mean(tf.log(sum2_samples)) --> -0.259 --> nan














##############################################################################
##############################################################################
# ALTERNATIVE
#
# logits_c = tf.reduce_sum(logits_true_class, axis=2, keep_dims=True)
# logits_diff = logits - logits_c
#
# print('logits_c ' + str(logits_c.shape))
# print(sess.run(logits_c))
# print('#' * 40)
#
# print('logits_diff ' + str(logits_diff.shape))
# print(sess.run(logits_diff))
# print('#' * 40)
#
# # exclude ones (exp(0) for true class - true class) by setting to zero before summing
# logits_diff_exp = tf.multiply(tf.exp(logits_diff), (label_one_hot - 1) * (-1))
#
# print('logits_diff_exp ' + str(logits_diff_exp.shape))
# print(sess.run(logits_diff_exp))
# print('#' * 40)
#
# logits_sum1_class = tf.reduce_sum(logits_diff_exp, axis=2, keep_dims=True)
#
# print('logits_sum1_class ' + str(logits_sum1_class.shape))
# print(sess.run(logits_sum1_class))
# print('#' * 40)
#
# log_sum1 = tf.log(logits_sum1_class)
#
# print('log_sum1 ' + str(log_sum1.shape))
# print(sess.run(log_sum1))
# print('#' * 40)
#
# # sum over samples (with mean to normalize)
# sum2 = tf.reduce_mean(log_sum1, axis=0)
#
# print('sum2 ' + str(sum2.shape))
# print(sess.run(sum2))
# print('#' * 40)
#
# # sum over pixels to form scalar (with mean to normalize)
# loss = tf.reduce_mean(sum2)
#
# print('loss ' + str(loss.shape))
# print(sess.run(loss))
# print('#' * 40)


#sum1 = tf.reduce_sum(logits_other_classes, axis=2, keep_dims=True)


##############################################################################
##############################################################################
# FIRST VERSION IN MODEL TRAIN OP

# # created [aleatoric_sample_n] of corrupted logits
# sampled_logits = sample_corrupt_logits(logits, sigma_activations, self.aleatoric_sample_n)  # [samples, batch_size, x, y, classes]
#
# # mask with one_hot_labels (but keep dims)
# logits_true_class = tf.multiply(sampled_logits, label_one_hot)
# logits_other_classes = tf.multiply(sampled_logits, label_one_hot_negated)
#
# # reduced since all other entries are 0 (because of one_hot_label_mask)
# true_class = tf.reduce_sum(logits_true_class, axis=-1, keep_dims=True)
# # sum over other classes
# sum1_other_classes = tf.log(tf.reduce_sum(tf.exp(logits_other_classes), axis=-1, keep_dims=True))
# # sum over and divide by number of samples
# sum2_samples = tf.reduce_mean(sum1_other_classes - true_class, axis=0)
# # sum over and divide by number of pixels
# loss_aletaoric = tf.reduce_mean(sum2_samples)

#
# # created [aleatoric_sample_n] of corrupted logits
# sampled_logits = sample_corrupt_logits(logits, sigma_activations, self.aleatoric_sample_n)  # [samples, batch_size, x, y, classes]
# # squash with softmax for numerical stability
#
# # reduced since all other entries are 0 (because of one_hot_label_mask)
# true_class = tf.reduce_sum(logits_true_class, axis=-1, keep_dims=True)
# # sum over other classes
# sum1_other_classes = tf.log(tf.reduce_sum(tf.exp(logits_other_classes), axis=-1, keep_dims=True))
# # sum over and divide by number of samples
# sum2_samples = tf.reduce_mean(sum1_other_classes - true_class, axis=0)
# # sum over and divide by number of pixels
# loss_aletaoric = tf.reduce_mean(sum2_samples)
# sampled_logits = tf.nn.softmax(sampled_logits, dim=2)
#
# # mask with one_hot_labels (but keep dims)
# logits_true_class = tf.multiply(sampled_logits, label_one_hot)
# logits_other_classes = tf.multiply(sampled_logits, label_one_hot_negated)
