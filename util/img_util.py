# IMAGE UTILITIES
# =================
#
# This code was adapted and partially modified from the tf_unet github (https://github.com/jakeret/tf_unet).
# All rights remain with the original author.
#
# License: GNU General Public License, see <http://www.gnu.org/licenses/>
# Author: jakeret       |   Modifications: Hannes Horneber
# Date: 2018-03-18


'''
Created on Aug 10, 2016

author: jakeret, changes/additions: Hannes Horneber
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
from scipy import stats
from math import log as logarithm

def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)
    
    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)
    
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()

def img_info(img, batch = False):
    n_channels = img.shape[-1] if (len(img.shape) > 2 + (1 if batch else 0)) else 1
    min_max = []
    if n_channels > 1:
        for c in range(n_channels):
            min_max.append( (np.min(img[..., c]),np.max(img[..., c])) )
    else: min_max.append( (np.min(img),np.max(img)) )
    type = img.dtype

    info = "info: %s %s, \nchannelwise min_max= %s" % (str(img.shape), str(type), str(min_max))
    if img.dtype != np.float32: info = info + "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
    return info

def to_rgb_heatmap(img, rgb_256=True):
    """
    Converts the given array NxM array into a NxMx3 RGB image. If the number of channels is not
    1 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny], e.g. img = [[0.9, 0.3], [0.2, 0.1]]

    :returns img: the rgb image [nx, ny, 3]
    """
    # logging.debug('#################################################################################################')
    # logging.debug(stats.describe(img.flatten()))
    img = np.squeeze(img)  # remove "empty" dims
    cmap = plt.get_cmap('jet')  # get colormap
    # logging.debug('img: ' + str(img.shape) + ' | min_channel_0 ' + str( np.amin(img[... , 0 ]))
    #               + ' | max_channel_0 ' + str(np.amax(img[..., 0])))

    # the cmap __call__ generates a rgba image
    rgba_img = cmap(img)
    # logging.debug('rgba_img size: ' + str(rgba_img.shape))
    rgb_img = np.delete(rgba_img, 3, 2)

    # logging.debug(stats.describe(rgb_img[:, :, 0].flatten()))
    # logging.debug(stats.describe(rgb_img[:, :, 1].flatten()))
    # logging.debug(stats.describe(rgb_img[:, :, 2].flatten()))

    # rescale images to [0,255)
    if rgb_256:
        rgb_img[np.isnan(rgb_img)] = 0
        if np.amin(rgb_img) < 0: rgb_img -= np.amin(rgb_img)
        rgb_img /= np.amax(rgb_img)
        rgb_img *= 255
    return rgb_img


def to_rgb(img, normalize=False, rgb256=True):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) and the array is cast to uint8
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img) # makes sure channels is at least 1
    channels = img.shape[-1]
    if channels == 1:
        img = np.tile(img, 3)
    elif channels == 2: # unexpected, but handle it by adding a zeroes channel
        img = np.concatenate(np.zeros([img.shape[0], img.shape[1], 1], dtype=img.dtype) , axis=2)
    else:  # channels > 3, reduce to 3 channels
        img = img[... , 0:3]

    img[np.isnan(img)] = 0  # remove NaN
    if np.amin(img) < 0 or normalize: img -= np.amin(img)         # level to zero
    if np.amax(img) > 1 or normalize: img = img / np.amax(img)    # scale to 0 - 1    (img /= np.amax(img))
    if rgb256 and np.amax(img) <= 1:
        img *= 255 # scale to 0 - 255
        img = img.astype(np.uint8)
    return img

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border
    (expects a tensor of shape [batches, nx, ny, ...]).
    
    :param data: the array to crop
    :param shape: the target shape
    """
    if data.shape[1] == shape[1] and data.shape[2] == shape[2]:
        return data # do nothing, already in shape

    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]


def crop_to_shape_2(data, shape):
    """
    Crops the array to the given image shape by removing the border
    Can handle a tensor of shape [batches, nx, ny, channels] or [nx, ny, channels].
    For the latter it expects square images!

    :param data: the array to crop
    :param shape: the target shape
    """
    if len(shape) == 4:
        offset0 = (data.shape[1] - shape[1]) // 2
        offset1 = (data.shape[2] - shape[2]) // 2
        return data[:, offset0:(-offset0), offset1:(-offset1)]
    else:
        offset0 = (data.shape[0] - shape[0]) // 2
        offset1 = (data.shape[1] - shape[1]) // 2
        return data[offset0:(-offset0), offset1:(-offset1)]


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image.
    Expects array of shape [batches, nx, ny, channels]
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img

def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """

    Image.fromarray(img.round().astype(np.uint8)).save(path, dpi=[300,300], quality=90)


def entropy2_x(labels):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        ent -= i * (logarithm(i)/logarithm(n_classes))

    return ent


