import tensorflow as tf
import numpy as np
import math

def softmax_stable(logits, axis):
    # subtract maximum to avoid too large exponents
    exps = tf.exp(logits - tf.reduce_max(logits, axis=axis))
    softmax = tf.divide(exps, tf.reduce_sum(exps, axis=axis))
    return softmax

def regularize_sigma(sigma):
    # scale sigma_activations
    return tf.nn.sigmoid(sigma)
    #return tf.multiply(sigma_activations, 0)


def sample_corrupt_logits(logits, sigma, sample_n):
    """
    returns tensor with sample_n corrupted logit tensors
    :param logits: tensor with logits [batch_size, x, y, classes]
    :param sigma: variance tensor, needs to be broadcastable to logit shape
    [batch_size, x, y, classes] or [1, x, y, classes] or [1, x, y, 1]
    :param sample_n: int - number of samples per logit
    :return: tensor with sampled corrupted logits [sample_n, batch_size, x, y, classes]
    """
    # this makes use of N(0,sigma_activations**2) = N(0,1) * sigma_activations
    # for easy computation of the corruption
    new_shape = [sample_n, tf.shape(logits)[0], tf.shape(logits)[1],
                 tf.shape(logits)[2], tf.shape(logits)[3]]
    # generating standard noise
    # to use a different distribution, see:
    # https://www.tensorflow.org/api_docs/python/tf/distributions/Laplace#sample
    gaussian = tf.random_normal(new_shape, mean=0.0, stddev=1.0, dtype=logits.dtype)
    # "weighting" noise with sigma_activations
    noise = tf.multiply(gaussian, sigma)
    # adding noise to logits
    return tf.add(logits, noise)


def aleatoric_entropy(logits, sigma_activations, n_samples):
    """

    :param logits:
    :param sigma_activations:
    :param n_samples:
    :return: entropy, sampled_logits_sm
    """
    sigma = regularize_sigma(sigma_activations)

    # create N=[aleatoric_sample_n] of corrupted logits
    sampled_logits = sample_corrupt_logits(logits, sigma, n_samples) # [samples, batch_size, x, y, classes]
    # mean of samples
    sampled_logits_mean = tf.reduce_mean(sampled_logits, axis=0)
    # squash logits vector with softmax, dim = class_dim
    sampled_logits_sm = tf.nn.softmax(sampled_logits_mean, dim=-1)

    # entropy of mean of sampled logits
    # per pixel entropy = - SUM_C [ p_c * log(p_c) ] with classes C = [0, ..., C]
    entropy = tf.multiply(tf.reduce_sum(tf.multiply(sampled_logits_sm, tf.log(sampled_logits_sm)), axis=-1), -1)
    return entropy, sampled_logits_mean


def aleatoric_cross_entropy(logits, label_one_hot, sigma_activations, n_samples, regularization = None):
    """
    Calculates aleatoric loss by sampling over corrupted logits.

    :param logits: network output [batch_size, x, y, classes]
    :param label_one_hot: groundtruth [batch_size, x, y, classes]
    :param sigma: learned variance [batch_size, x, y, classes] or [1, x, y, classes] or [1, x, y, 1]
    :return:
    """
    sigma = regularize_sigma(sigma_activations)

    # create N=[aleatoric_sample_n] of corrupted logits
    sampled_logits = sample_corrupt_logits(logits, sigma, n_samples) # [samples, batch_size, x, y, classes]
    # squash logits vector with softmax, dim = class_dim
    sampled_logits = tf.nn.softmax(sampled_logits, dim=-1)

    # mask the tensor with one_hot_labels (but keep dims)
    sampled_logits_true_class = tf.multiply(sampled_logits, label_one_hot)
    # reduce since all other entries are 0 (because of one_hot_label_mask)
    sampled_logits_true_class = tf.reduce_sum(sampled_logits_true_class, axis=-1, keep_dims=True)

    # add epsilon for numerical stability
    sampled_cross_entropy = tf.multiply( tf.log(sampled_logits_true_class + tf.constant(1.0e-4, dtype=tf.float32)), -1)

    # sum over and divide by number of pixels
    # loss_per_pixel = tf.reduce_mean(sampled_cross_entropy)
    # sum over and divide by number of samples
    # loss_per_pixel = tf.divide(tf.reduce_sum(sampled_cross_entropy, axis=0), n_samples)
    # do both in one step:
    aleatoric_CE_mean = tf.reduce_mean(sampled_cross_entropy, axis=0)

    if regularization is not None:
        if regularization == 'mean_square':
            regularization_term = tf.reduce_mean(tf.square(sigma))
        elif regularization == 'square':
            regularization_term = tf.square(sigma)
        else:
            regularization_term = 0 # no default

            aleatoric_CE_mean = aleatoric_CE_mean + regularization_term

    return aleatoric_CE_mean, sigma, sampled_logits



def gaussian_entropy(sigma):
    """
    Entropy of Gaussian
    Formula from: https://en.wikipedia.org/wiki/Normal_distribution
    1/2 * log(2*pi*e*sigma_activations**2)
    """
    return tf.divide(tf.multiply(tf.square(sigma), (2*math.pi*math.e)), 2)


def epistemic_entropy(sampled_softmax):
    """

    :param sampled_softmax_mean: [n_samples, x, y, n_classes]
    :return: entropy, sampled_softmax_mean, prediction
    """
    # mean of samples
    sampled_softmax_mean = np.mean(sampled_softmax, axis=0)
    # squash softmax mean with argmax to get segmentation, dim = class_dim
    prediction = np.argmax(sampled_softmax_mean, axis=-1)

    # entropy of mean of sampled softmaxed logits
    # per pixel entropy = - SUM_C [ p_c * log(p_c) ] with classes C = [0, ..., C]
    entropy = - np.sum(sampled_softmax_mean * np.nan_to_num(np.log(sampled_softmax_mean)), axis=-1)
    return entropy, sampled_softmax_mean, prediction