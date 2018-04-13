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

#logits = tf.constant([[.1, .2, .7],
                      # [.3, .4, .3]])
# label_one_hot = tf.constant([[0, 0, 1],
#                              [0, 1, 0]])

# for better console visualization, class_axis = 1
#logits = tf.random_uniform((1, 3, 4, 4), seed=2)

T_samples = 2

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
print('#' * 40)

def corrupt_logits(logits, sigma):
    gaussian = tf.random_normal(tf.shape(logits), mean=0.0, stddev=1.0, dtype=logits.dtype)
    noise = tf.multiply(gaussian, sigma)  #
    return tf.add(logits, noise)

def sample_corrupt_logits(logits, sigma, sample_n):
    new_shape = [sample_n, tf.shape(logits)[0],tf.shape(logits)[1], tf.shape(logits)[2], tf.shape(logits)[3]]
    gaussian = tf.random_normal(new_shape, mean=0.0, stddev=1.0, dtype=logits.dtype)
    noise = tf.multiply(gaussian, sigma)  #
    return tf.add(logits, noise)

sampled_logits = corrupt_logits(logits, sigma)

print('corrupted_logits ' + str(sampled_logits.shape))
print(sess.run(sampled_logits))
print('#' * 40)


sampled_logits = sample_corrupt_logits(logits, sigma, T_samples)

print('sampled_corrupted_logits ' + str(sampled_logits.shape))
print(sess.run(sampled_logits))
print('#' * 40)

#print('logits_softmax ' + str(logits_softmax.shape))
#logits_softmax = tf.nn.softmax(logits, dim=1 )
#print(sess.run(logits_softmax))

#print('logits_maxed ' + str(logits_maxed.shape))
#logits_maxed = tf.reduce_max(logits_soft, axis=1)
#print(sess.run([logits_soft, logits_maxed]))

logits = sampled_logits
print('#' * 20)

#######################################

#label_one_hot = tf.cast(label_one_hot, tf.bool)# [batch_size, x, y, classes]
#logits_true_class = tf.boolean_mask(logits, label_one_hot)
logits_true_class = tf.multiply(logits, label_one_hot)
#logits_other_classes = tf.boolean_mask(logits, tf.logical_not(label_one_hot))
logits_other_classes = tf.multiply(logits, (label_one_hot - 1)*(-1))

print('logits_true_class ' + str(logits_true_class.shape))
print(sess.run(logits_true_class))
print('logits_other_classes ' + str(logits_other_classes.shape))
print(sess.run(logits_other_classes))
print('#' * 40)

logits_true_class_rs = tf.reduce_sum(logits_true_class, axis=2, keep_dims=True)
logits_other_classes_rs = tf.log(tf.reduce_sum(tf.exp(logits_other_classes), axis=2, keep_dims=True))

print('logits_true_class_rs ' + str(logits_true_class_rs.shape))
print(sess.run(logits_true_class_rs))
print('logits_other_classes_rs ' + str(logits_other_classes_rs.shape))
print(sess.run(logits_other_classes_rs))
print('#' * 40)

#print(sess.run([logits, logits_true_class_rs]))

#######################################

in_sum_t = logits_other_classes_rs - logits_true_class_rs

print('in_sum_t ' + str(in_sum_t.shape))
print(sess.run(in_sum_t))
# print('logits_other_classes_rs ' + str(logits_other_classes_rs.shape))
# print(sess.run(logits_other_classes_rs))
print('#' * 40)

#######################################

sum_t = tf.divide(tf.reduce_sum(in_sum_t, axis=0), T_samples)

print('sum_t ' + str(sum_t.shape))
print(sess.run(sum_t))
# print('logits_other_classes_rs ' + str(logits_other_classes_rs.shape))
# print(sess.run(logits_other_classes_rs))
print('#' * 40)







#V = logits[b, x, y, c]

# [[['b0x0y0c0', 'b0x1y0c0']['b0x0y0c1' 'b0x1y0c1'
# ['b0x0y1c0' 'b0x1y1c0']]   'b0x0y1c1' 'b0x1y1c1']
#
#  ['b1x0y0c0' 'b1x1y0c0'['b1x0y0c1' 'b1x1y0c1'
#                         'b1x0y1c0' 'b1x1y1c0']   'b1x0y1c1' 'b1x1y1c1']]
#
# labels[b, x, y, c]