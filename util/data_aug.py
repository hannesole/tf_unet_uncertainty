# AUGMENTATION LIB
# ================
# Functions to augment images (and their corresponding masks).
#
# Author: Hannes Horneber
# Date: 2018-03-20

import imgaug as ia
from imgaug import augmenters as iaa
from timeit import default_timer as timer
import logging
import numpy as np
import random

from util import img_util

def augment(img, label, weights, p=0.5):
    '''

    :param img:
    :param label:
    :param weights:
    :return:
    '''
    p = 0.75

    # preserve dtypes when augmenting!
    original_dtypes = [img.dtype, label.dtype, weights.dtype ]

    #TODO remove debug code...
    debug = False
    if debug: start = timer()
    if debug: logging.debug("start augment: img %s %s, label %s %s, weights %s %s"
                            % (str(img.shape), str(img.dtype),
                               str(label.shape), str(label.dtype),
                               str(weights.shape), str(weights.dtype)))
    if debug: logging.debug("img: " + img_util.img_info(img))
    # if debug: logging.debug("label: " + img_util.img_info(label))
    # if debug: logging.debug("weights: " + img_util.img_info(weights))


    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases, e.g. Sometimes(0.5, GaussianBlur(0.3))
    sometimes = lambda aug: iaa.Sometimes(p, aug)

    # define sequence for positional transformation augmentations
    # (unlike value augmentation, these augs need to be applied to corresponding masks (labels, weights) as well)
    seq_pos = iaa.Sequential(
        [
            iaa.Fliplr(p/2), # horizontally flip 25% of all images
            iaa.Flipud(p/2), # vertically flip 25% of all images

            sometimes(iaa.OneOf([ # either some affine transformations, a piecewise affine or a perspective transform
                iaa.Affine( # might want to split out single transforms with iaa.SomeOf((0, 3), [])
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axi
                            rotate=(0,360), # rotate random degrees
                            shear=(-16, 16), # shear by -16 to +16 degrees
                            order=[0], # use nearest neighbour or bilinear interpolation (fast)
                            cval=0, # if mode is constant, use 0 (black) as constant value (for padding)
                            mode="reflect" # for new pixels use vector mirrored on the first and last values of the vector along each axis
                        ),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.04)))  # sometimes move parts of the image around
                #sometimes(iaa.PerspectiveTransform(scale=(0.02, 0.08)))    # might change dtype to uint8!
            ]))
        ],
        random_order=True
    )

    # convert stochastic sequence of augs to deterministic, so that augs are reproducible (for mask and weights)
    seq_pos_det = seq_pos.to_deterministic()

    # apply (deterministic!) positional transformation to image and masks
    img_aug = seq_pos_det.augment_image(img)
    label_aug = seq_pos_det.augment_image(label)
    weights_aug = seq_pos_det.augment_image(weights)


    # apply additional value augmenation to image only

    # define sequence for value augmentation
    # seq_val = iaa.Sequential(
    #     [
    #           sometimes(iaa.ContrastNormalization((0.75, 1.25), per_channel=0.5)),  # enhance or flatten the contrast
    #           # sometimes(iaa.Add((-10, 10), per_channel=0.5)), # not sure whether (0, 255) or (0, 1) images -> use multiply, more robust
    #           sometimes(iaa.Multiply((0.75, 1.25), per_channel=True))   # enhance or dampen single channels
    #     ]
    # )
    #img_aug = seq_pos.augment_images(img_aug)

    # sometimes custom per channel augmentation:
    if random.random() < p:
        multiplicative = random.random() < 0.5    # randomly apply additive, else multiplicative 50%
        per_channel = (random.random() < 0.5)     # randomly apply per channel 50%
        if not per_channel:
            if multiplicative:
                rn = random.uniform(0.75, 1.25)
                img_aug = img_aug * rn  # augment multiplicative
            else:
                rn = random.uniform(-0.1, 0.1)
                img_aug = img_aug + rn  # augment additive

            # make sure resulting values are between [0, 1]
            if rn > (1 if multiplicative else 0):
                img_aug = np.minimum(img_aug, np.ones(img_aug.shape))
            elif rn < (1 if multiplicative else 0):
                img_aug = np.maximum(img_aug, np.zeros(img_aug.shape))
        else: # do everything channelwise
            for c in range(img_aug.shape[-1]):
                if multiplicative:
                    rn = random.uniform(0.9, 1.1)
                    img_aug[..., c] = img_aug[..., c] * rn # augment multiplicative
                else:
                    rn = random.uniform(-0.05, 0.05)
                    img_aug[..., c] = img_aug[..., c] + rn # augment additive

                # make sure resulting values are between [0, 1]
                if rn > (0 if multiplicative else 1):
                    img_aug[..., c] = np.minimum(img_aug[..., c], np.ones(img_aug[..., c].shape))
                elif rn < (0 if multiplicative else 1):
                    img_aug[..., c] = np.maximum(img_aug[..., c], np.zeros(img_aug[..., c].shape))

        if debug: logging.debug("value aug: %s %s " % (("multiplicative" if multiplicative else "additive"),
                                                        ("channelwise" if per_channel else "on image with " + str(rn))))


    if debug: logging.debug("img_aug: " + img_util.img_info(img_aug))
    # if debug: logging.debug("label_aug: " + img_util.img_info(label_aug))
    # if debug: logging.debug("weights_aug: " + img_util.img_info(weights_aug))
    if debug: end = timer()
    if debug: logging.debug("augmented in %.4f s : : img %s %s, label %s %s, weights %s %s"
                            % ((end - start),
                               str(img_aug.shape), str(img_aug.dtype),
                               str(label_aug.shape), str(label_aug.dtype),
                               str(weights_aug.shape), str(weights_aug.dtype)))


    return img_aug.astype(original_dtypes[0]), \
        label_aug.astype(original_dtypes[1]), \
        weights_aug.astype(original_dtypes[2])

# data augmentation from CIFAR10
def _preproc_Augment(img, label, debug=False):
    if debug:
        tf.summary.image('image', img)


    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop()  (img,
                                     [output_height, output_width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    if add_image_summaries:
        tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.


    return tf.image.per_image_standardization(distorted_image)

    return img, label