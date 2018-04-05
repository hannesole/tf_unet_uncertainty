# TFRECORD DATA LIB
# =================
# These functions serve to build a tf_records file from .tif files
# (should be compatible but isn't tested for other image formats)
# and to test the resulting dataset.
#
# Author: Hannes Horneber
# Date: 2018-03-18

import tensorflow as tf
import cv2
import numpy as np
import logging
import os
import glob
import matplotlib.pyplot as plt
from random import shuffle
from unet import data_layers


###################################################
# LOCAL FUNCTIONS
# ===============

def _find_img_and_label_files(search_path, mask_suffix='_mask', img_suffix=''):
    # expects a folder with files img as img and img_mask (e.g. tile01.tif, tile01_mask.tif)
    # if naming is otherwise, specify mask_suffix / img_suffix
    all_files = glob.glob(search_path)
    label_files = [name for name in all_files if mask_suffix in name]
    img_files = [name.replace(mask_suffix, img_suffix) for name in label_files if mask_suffix in name]
    return img_files, label_files

def load_image(file, resize_shape=None, color=True):
    # read an image and resize to (1024, 1024)
    logging.debug("load_image: %s " % (file))
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    logging.debug("          > shape %s (requested resize: %s - perform:  %s )"
                  % (str(img.shape), (str(resize_shape) if resize_shape is not None else 'None'),
                     (str(resize_shape[0] != img.shape[0]) if resize_shape is not None else 'No') ) )

    if resize_shape is not None and (resize_shape[0] != img.shape[0] or resize_shape[1] != img.shape[1]):
        img = cv2.resize(img, (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_CUBIC)
        logging.debug("          > resized to %s " % (str(img.shape)))

    # cv2 loads color images by default as BGR, convert it to RGB
    if color and (len(img.shape) == 3) and (img.shape[2] == 3): img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 2:
        img[img != 0] = 1 # make binary mask for grayscale images
        #logging.debug("%s" % (img))

    img = img.astype(np.float32)
    return img, img.shape

# create tf int64 feature from values (probably not needed here)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# create tf bytes feature from values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


###################################################


def build_tf_records(datasource, destination, resize_shape=None, shuffle_data=True):
    # READ AND ARRANGE FILELISTS
    # --------------------------
    logging.info('build_tf_records >> reading from: ' + datasource)
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
        filename = destination + os.sep + i_mode + '.tfrecords' # address to save the TFRecords file
        logging.info('build_tf_records >> writing to: ' + filename)

        files_img_mode = {
            'train': train_files_img,
            'val': val_files_img,
            'test': test_files_img
        }[i_mode]
        files_labels_mode = {
            'train': train_files_labels,
            'val': val_files_labels,
            'test': test_files_labels
        }[i_mode]

        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(len(files_img_mode)):
            # print how many images are saved every 4 images
            if not (i+1) % 4: logging.info(i_mode + '  write: {}/{}'.format(i+1, len(files_img_mode)))

            # Load the image
            img, shape_img = load_image(files_img_mode[i], resize_shape=resize_shape)
            label, shape_label = load_image(files_labels_mode[i], resize_shape=resize_shape)
            #TODO implement reading weight files
            # shape_weight = [shape_img[0], shape_img[1]]
            # weight = np.ones(shape_weight, np.float32)
            #'data/weight': _bytes_feature(tf.compat.as_bytes(weight.tostring()))

            # An Example protocol buffer contains Features.
            # Feature is a protocol to describe the img and could have three types: bytes, float, and int64.
            # Create a feature
            feature = {'data/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'data/label': _bytes_feature(tf.compat.as_bytes(label.tostring()))}

            # Before we can store the img into a TFRecords file, we should stuff it in a protocol buffer called Example.
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
    logging.debug("shape_img: %s | shape_label: %s" %(str(shape_img), str(shape_label)))
    return shape_img, shape_label



def test_tf_records(data_path, mode='train', shape_img=[1024, 1024, 3], shape_label=[1024, 1024, 1],
                    batch_size = 10, shuffle=True, output_dir=None, n_samples=10 ):
    # create filename from data_path and mode
    data_path = data_path + os.sep + mode + '.tfrecords'
    logging.debug("read_tf_records >> %s | %s from %s" % (str(shape_img), str(shape_label), data_path))

    imgs, labels, _ = data_layers.data_TFRqueue(data_path, is_training=True,
                                                shape_img=shape_img, shape_label=shape_label, batch_size=batch_size)

    with tf.Session() as sess:
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Some functions of tf.train such as tf.train.shuffle_batch add tf.train.QueueRunner objects to your graph.
        # Each of these objects hold a list of enqueue op for a queue to run in a thread.
        # Therefore, to fill a queue you need to call tf.train.start_queue_runners
        # which starts threades for all the queue runners in the graph.
        # However, to manage these threads you need a tf.train.Coordinator to terminate the threads at the proper time.

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # write batches as graph
        if output_dir is not None:
            os.mkdir(output_dir)

            for batch_n in range(n_samples):
                img_batch, label_batch = sess.run([imgs, labels])
                #logging.debug(" label_batch %s" % (str(label_batch)))
                img_batch = img_batch.astype(np.uint8)
                label_batch = label_batch.astype(np.uint8)
                for j in range(batch_size):
                    logging.debug("   img: %s/%s | shape: %s | %s"
                                  % (str(j), str(batch_size), str(img_batch[j, ...].shape), str(label_batch[j, ...].shape)))
                    plt.subplot(2, batch_size, (j*2) + 1)
                    plt.imshow(img_batch[j, ...])
                    plt.subplot(2, batch_size, (j*2) + 2)
                    plt.imshow(label_batch[j, ...])
                    #plt.title('cat' if label[j] == 0 else 'dog')
                logging.info(' read_tf_records >> writing batch to: ' + (output_dir + os.sep + 'tile_' + str(batch_n) + '.png'))
                plt.savefig(output_dir + os.sep + 'batch_' + str(batch_n) + '.png')
                #plt.close(fig)

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()



