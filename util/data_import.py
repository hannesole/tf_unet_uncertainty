# DATA IMPORT
# =============
# Functions to find and read image data stored on disk.
#
# Author: Hannes Horneber
# Date: 2018-03-18


import cv2
import numpy as np
import logging
import os
import glob
import matplotlib.pyplot as plt
from random import shuffle

def _find_img_and_label_files(search_path, mask_suffix='_mask', img_suffix=''):
    # expects a searchpath (e.g. /dir/*.tif) with files img as img and img_mask (e.g. tile01.tif, tile01_mask.tif)
    # if naming is otherwise, specify mask_suffix / img_suffix
    all_files = glob.glob(search_path)
    label_files = [name for name in all_files if mask_suffix in name]
    img_files = [name.replace(mask_suffix, img_suffix) for name in label_files if mask_suffix in name]
    return img_files, label_files

def _find_img_files(search_path, img_suffix='.tif'):
    # expects a searchpath (e.g. /dir/*.tif) with img files img (e.g. tile01.tif)
    # if format/naming is otherwise, specify img_suffix or use a regex for search path
    # e.g. search_path = /data/train/folder_with_images_and_other_files/*.tif"
    all_files = glob.glob(search_path)
    img_files = [name for name in all_files if img_suffix in name]
    return img_files

def load_image(file, resize_shape=None, RGB=True, FLOAT01=True, binarize_gray=False):
    # read an image and optionally resize to resize_shape
    logging.debug("load_image: %s " % (file))
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    logging.debug("          > shape %s (requested resize: %s - perform resize:  %s )"
                  % (str(img.shape), (str(resize_shape) if resize_shape is not None else 'None'),
                     (str(resize_shape[0] != img.shape[0]) if resize_shape is not None else 'No') ) )

    if resize_shape is not None and (resize_shape[0] != img.shape[0] or resize_shape[1] != img.shape[1]):
        img = cv2.resize(img, (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_CUBIC)
        logging.debug("          > resized to %s " % (str(img.shape)))

    # cv2 loads RGB images by default as BGR, convert it to RGB
    if RGB and (len(img.shape) == 3) and (img.shape[2] == 3): img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if binarize_gray and (len(img.shape) == 2):
        img[img != 0] = 1 # make binary mask for grayscale images
    else:
        if FLOAT01: img = img.astype(np.float32) / 255
    return img

def create_img_provider(src_dir, loop=False, shuffle_data=False):
    img_list = _find_img_files(src_dir)
    if shuffle_data: img_list = shuffle(img_list)

    img_i = 0
    while(loop or img_i < len(img_list)):
        img_i = img_i + 1
        yield load_image(img_list[img_i])



def create_img_batch_provider(src_dir, batch_size, loop=False, shuffle_data=False, no_unicolor=True):
    img_list = _find_img_files(src_dir)
    if shuffle_data: shuffle(img_list)

    img_i = 0
    while(loop or img_i < len(img_list)):
        for j in range(batch_size):
            img_i = img_i + j
            if j == 0:  # create img_batch
                img_batch = load_image(img_list[img_i])[np.newaxis, ...]
            else:       # append (concatenate) to img_batch
                np.concatenate((img_batch, load_image(img_list[img_i])[np.newaxis, ...]), axis=0)
        yield img_batch


def build_sets(datasource, destination, resize_shape=None, shuffle_data=True):
    # READ AND ARRANGE FILELISTS
    # --------------------------
    logging.info('build_sets >> reading from: ' + datasource)
    # read addresses and labels from the 'train' folder
    files_img, files_labels = _find_img_and_label_files(datasource)
    logging.info('found files: ' + str(files_img))

    # to shuffle img
    if shuffle_data:
        combined = list(zip(files_img, files_labels))
        logging.info('list(zip(files_img, files_labels)): ' + str(combined))
        shuffle(combined)
        logging.info('shuffled: ' + str(combined))
        files_img[:], files_labels[:] = zip(*combined)

    # Divide into 60% train, 20% validation, and 20% test
    img_perc = 0.6
    val_perc = 0.2  # the rest is test img
    assert (img_perc + val_perc) < 1
    train_files_img = files_img[0:int(img_perc * len(files_img))]
    train_files_labels = files_labels[0:int(img_perc * len(files_labels))]
    val_files_img = files_img[int(img_perc * len(files_img)):int((img_perc + val_perc) * len(files_img))]
    val_files_labels = files_labels[int(img_perc * len(files_img)):int((img_perc + val_perc) * len(files_img))]
    test_files_img = files_img[int((img_perc + val_perc) * len(files_img)):]
    test_files_labels = files_labels[int((img_perc + val_perc) * len(files_labels)):]
    logging.debug(
        "train_files_img: %s | train_files_labels: %s" %(str(len(train_files_img)), str(len(train_files_labels))))
    logging.debug(
        "val_files_img: %s | val_files_labels: %s" % (str(len(val_files_img)), str(len(val_files_labels))))
    logging.debug(
        "test_files_img: %s | test_files_labels: %s" % (str(len(test_files_img)), str(len(test_files_labels))))

    # WRITE TFRECORDS FROM FILELISTS
    # ------------------------------
    mode = ['train', 'val', 'test']
    for i_mode in mode:
        filename = destination + os.sep + '.tfrecords' # address to save the TFRecords file
        logging.info('build_sets >> writing to: ' + filename)

        files_img_mode = {      # read like a switch statement of [i_mode]
            'train': train_files_img,
            'val': val_files_img,
            'test': test_files_img
        }[i_mode]
        files_labels_mode = {   # read like a switch statement of [i_mode]
            'train': train_files_labels,
            'val': val_files_labels,
            'test': test_files_labels
        }[i_mode]

        # open the set
        for i in range(len(files_img_mode)):
            # print how many images are saved every 4 images
            if not (i+1) % 4: logging.info(i_mode + '  write: {}/{}'.format(i+1, len(files_img_mode)))

            # Load the image
            img, shape_img = load_image(files_img_mode[i], resize_shape=resize_shape)
            label, shape_label = load_image(files_labels_mode[i], resize_shape=resize_shape)
            #TODO implement reading weight files
            # shape_weight = [shape_img[0], shape_img[1]]

    logging.debug("shape_img: %s | shape_label: %s" %(str(shape_img), str(shape_label)))
    return shape_img, shape_label
