import tensorflow as tf


def sample_corrupt_logits(logits, sigma, sample_n):
    """
    returns tensor with sample_n corrupted logit tensors
    :param logits: tensor with logits [batch_size, x, y, classes]
    :param sigma: variance tensor, needs to be broadcastable to logit shape
    [batch_size, x, y, classes] or [1, x, y, classes] or [1, x, y, 1]
    :param sample_n: int - number of samples per logit
    :return: tensor with sampled corrupted logits [sample_n, batch_size, x, y, classes]
    """
    # this makes use of N(0,sigma**2) = N(0,1) * sigma
    # for easy computation of the corruption
    new_shape = [sample_n, tf.shape(logits)[0], tf.shape(logits)[1],
                 tf.shape(logits)[2], tf.shape(logits)[3]]
    # generating standard noise
    # to use a different distribution, see:
    # https://www.tensorflow.org/api_docs/python/tf/distributions/Laplace#sample
    gaussian = tf.random_normal(new_shape, mean=0.0, stddev=1.0, dtype=logits.dtype)
    # "weighting" noise with sigma
    noise = tf.multiply(gaussian, sigma)
    # adding noise to logits
    return tf.add(logits, noise)



def aleatoric_loss(logits, label_one_hot, sigma, n_samples):
    """
    Calculates aleatoric loss by sampling over corrupted logits.

    :param logits: network output [batch_size, x, y, classes]
    :param label_one_hot: groundtruth [batch_size, x, y, classes]
    :param sigma: learned variance [batch_size, x, y, classes] or [1, x, y, classes] or [1, x, y, 1]
    :return:
    """
    label_one_hot_negated = (label_one_hot - 1 ) *(-1) # [batch_size, x, y, classes]

    # create [aleatoric_samples] of corrupted logits
    sampled_logits = sample_corrupt_logits(logits, sigma, n_samples) # [samples, batch_size, x, y, classes]

    # squash logits vector with softmax
    #sampled_logits = tf.nn.softmax(sampled_logits, dim=-1)

    # mask the tensor with one_hot_labels (but keep dims)
    logits_true_class = tf.multiply(sampled_logits, label_one_hot)
    logits_other_classes = tf.multiply(sampled_logits, label_one_hot_negated)

    # reduce since all other entries are 0 (because of one_hot_label_mask)
    true_class = tf.reduce_sum(logits_true_class, axis=-1, keep_dims=True)

    # sum over other classes
    sum1_other_classes = tf.log(tf.reduce_sum(tf.exp(logits_other_classes), axis=-1, keep_dims=True))
    # sum over and divide by number of samples
    sum2_samples = tf.reduce_mean(tf.exp(true_class - sum1_other_classes), axis=0)
    # sum over and divide by number of pixels
    loss_aletaoric = tf.reduce_mean(tf.log(sum2_samples))

    # loss_aletaoric = tf.reduce_sum(tf.log(sum2_samples)) --> inf
    # loss_aletaoric = tf.reduce_mean(tf.log(sum2_samples)) --> -0.259 --> nan

    loss = loss_aletaoric
    return loss, sampled_logits