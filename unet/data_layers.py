# DATA LAYER
# ====================
#
# Contains implementations of different data layers:
#   data_tf_placeholder (feed_dict)
#   tf_records queue
#   hdf5_dataset queue
#   hdf5_table queue
#
# Queues are build using the QueueRunner API avoid the (presumingly slow) feed-dict method.
# (https://www.tensorflow.org/api_guides/python/reading_data).
#
# The implementations are partially tailored to my specific dataset with three parts
# (img-data, label-masks and weight-masks) and probably have to be adjusted for any other dataset.
#
# Author: Hannes Horneber
# Date: 2018-03-18

import tensorflow as tf
import numpy as np
import logging
import tftables     # for hdf5 reading
import h5py         # for hdf5 reading
import random

from timeit import default_timer as timer   # for debugging (timing)

from util import data_aug   # augmentation

# ##################################################################################
# ##############                     DATA LAYERS                        ############
# ##################################################################################


# #########      FEED DICT       #########
# ----------------------------------------
def data_tf_placeholder(shape_img, is_training=False, batch_size=None, shape_label=None, shape_weights=None):
    '''
    When running for training returns placeholders for input image batches, label batches and weight batches;
    for testing only input image batches.
    This requires using a feed_dict when running the model to fill the placeholders.

    :param shape_img: input image size, e.g. [1024, 1024, 3]
    :param shape_label: by default a single channel mask of image_shape dimensions [shape_img[0], shape_img[1]]
    :param shape_weights: by default a single channel mask of image_shape dimensions [shape_img[0], shape_img[1]]
    :param batch_size: (optional) if parameter is set, the placeholder will be static and batches MUST be sized accordingly
    :return: tf.placeholder object (for training multiple tf.placeholder objects).
    '''
    if is_training:
        if shape_label is None: shape_label = [shape_img[0], shape_img[1], 1]
        if shape_weights is None: shape_weights= [shape_img[0], shape_img[1], 1]
        return tf.placeholder(tf.float32, [batch_size, shape_img[0], shape_img[1], shape_img[2]]), \
               tf.placeholder(tf.float32, [batch_size, shape_label[0], shape_label[1], shape_label[2]]), \
               tf.placeholder(tf.float32, [batch_size, shape_weights[0], shape_weights[1], shape_weights[2]])
    else:
        return tf.placeholder(tf.float32, [batch_size, shape_img[0], shape_img[1], shape_img[2]]), \
               None, None # always return three arguments so that nr of expected arguments don't has to be changed


# #########      TFRECORDS       #########
# ----------------------------------------
def data_TFRqueue(data_path, shape_img, shape_label=None, shape_weights=None,
                  is_training=False, feature=None, shuffle=False, batch_size=1):
    '''
    Adds a TFRecord data layer to your model graph.
    The layer reads a file from path, decodes it's contents based
    including decode, reshape and batch_shuffle mechanics.
    This offers a pre-processing hook that can be used to transform data before feeding it to the network.

    To use this in your model, make sure you add the following commands when running your session:

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Some functions of tf.train such as tf.train.shuffle_batch add tf.train.QueueRunner objects to your graph.
        # Each of these objects holds a list of enqueue ops for a queue to run in a thread.
        # Therefore, to fill a queue you need to call tf.train.start_queue_runners,
        # which starts threads for all the queue runners in the graph.
        # To manage these threads you need a tf.train.Coordinator that terminates the threads at the proper time.
        # >> Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # /// other sess.run(...) steps here ///
        coord.request_stop() # Stop the threads
        coord.join(threads) # Wait for threads to stop
        sess.close()

    :param data_path: path to data
    :param is_training: by default False
    :param feature: (experimental! not working yet) how record is structured. By default with keys'data/image' and 'data/labels'
    :param shape_img: (optional) how data is shaped, by default shape_img=[1024, 1024, 3]
    :param shape_label: (optional) how label is shaped, by default shape_img=[1024, 1024, 1]
    :param shape_weights: by default a single channel mask of image_shape dimensions [shape_img[0], shape_img[1]]
    :param batch_size: (optional) by default 1
    :param shuffle: (optional) placeholder, not functional yet - batches will always be shuffled
    :return: if is_training: imgs, labels, weights (Tensors with batches, e.g [batch_size, 1024, 1024, channels],
    else (for testing) only imgs Tensor
    '''
    logging.debug("DataLayer TFRecords Queue >> %s | %s from %s" % (str(shape_img), str(shape_label), data_path))

    if shape_label is None: shape_label = [shape_img[0], shape_img[1], 1]
    if shape_weights is None: shape_weights = [shape_img[0], shape_img[1], 1]

    with tf.Session() as sess:
        # Create queue to hold and provide filenames of TFRecords
        #TODO change to support providing multiple filenames
        filename_queue = tf.train.string_input_producer([data_path])

        # Define a record reader that returns next records (serialized_strings) with read(filename_queue)
        tf_reader = tf.TFRecordReader()
        _, serialized_example = tf_reader.read(filename_queue)

        # define dictionary with feature keys (how serialized examples are decoded)
        # maps feature keys to FixedLenFeature or VarLenFeature values
        #TODO provide custom feature support (change here and when decoding)
        feature = {'data/image': tf.FixedLenFeature([], tf.string),
                   'data/label': tf.FixedLenFeature([], tf.string)}
        # ... then parse the record read by the reader
        # returns a dictionary which maps feature keys to Tensor values
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        # tf.decode_raw(bytes, out_type) takes a Tensor of type string and converts it to tf.float32 (typeout_type)
        img = tf.decode_raw(features['data/image'], tf.float32)
        label = tf.decode_raw(features['data/label'], tf.float32)

        # Reshape image data into the original shape
        img = tf.reshape(img, shape_img)
        label = tf.reshape(label, shape_label)

        #################################
        # PREPROCESS HOOK               #
        # Any pre-processing here ...   #
        #################################

        if shuffle:
            # Creates batches by randomly shuffling tensors / examples
             imgs, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size,
                                              capacity=30, num_threads=1,
                                              min_after_dequeue=1)
        else:
            # Creates unshuffled batches
            imgs, labels = tf.train.batch([img, label], batch_size=batch_size,
                                                  capacity=30, num_threads=1,
                                                  min_after_dequeue=1)

        #TODO weights are not yet in tf_records. Using dummy weights (all 1)
        weights = tf.ones(shape_weights, dtype=tf.float32, name='DummyWeights')

        if is_training: return imgs, labels, weights
        else: return imgs, None, None



def data_HDF5(data_path,
              shape_img, shape_label, shape_weights,
              is_training=False,
              batch_size=1, shuffle=False, augment=False,
              prefetch_n=None, prefetch_threads=None,
              resample_n=None,
              name=None):
    '''
    Adds a HDF5record data layer to your model graph.
    The layer reads from a single HDF5 datafile and expects dsets '/data' '/label' and '/weights' in the file.
    ...

    :param data_path: path to data
    :param is_training: (optional) by default False
    :param batch_size: (optional) by default 1
    :param shuffle: (optional) by default False
    :param prefetch_n: (optional) how many elements are prefeacjcby default 2*batch_size
    :return: a reader object that needs to be closed after running the session, the data
    '''

    with tf.variable_scope('DataSet_hdf5'):
        if prefetch_n is None: prefetch_n = batch_size * 2
        if prefetch_threads is None: prefetch_threads = 1
        logging.info("DataLayer HDF5 [%s] %s | prefetch %s w/ %s threads | source %s " %
                     ('' if name is None else name, ("with augmentation" if augment else ''),
                      str(prefetch_n), str(prefetch_threads), data_path))

        data_gen = hdf5_generator(data_path,
                           augment=augment,
                           repeat=(is_training or augment), cache_full_file=True, shuffle=shuffle,
                           resample_n=resample_n, name=name)

        sample_set = tf.data.Dataset.from_generator(
            data_gen,
            (tf.float32, tf.uint8, tf.float32),
            (tf.TensorShape(shape_img), tf.TensorShape(shape_label), tf.TensorShape(shape_weights))
        )

        #################################
        # TF PREPROCESS HOOK             #
        # Any tf pre-processing here ... #
        if augment:
            def py_aug_func(img, label, weights):
                # python augmentation wrapper for parallelization (otherwise could be done in data generator)
                img_aug, label_aug, weights_aug = tf.py_func(data_aug.augment,                   # py function
                                                             [img, label, weights],              # input to function
                                                             (tf.float32, tf.uint8, tf.float32)) # Tout (output Types)
                # applying py_func dels shape information. assume shapes are the same as input (make sure they are in py_func)
                img_aug.set_shape(img.get_shape())
                label_aug.set_shape(label.get_shape())
                weights_aug.set_shape(weights.get_shape())
                return img_aug, label_aug, weights_aug

            sample_set = sample_set.map(py_aug_func, num_parallel_calls=prefetch_threads)
        #################################

        # repeat and shuffle: https://www.tensorflow.org/versions/master/performance/datasets_performance#repeat_and_shuffle
        # in Tensorflow r1.4 tensorflow.contrib.data.shuffle_and_repeat is not available
        if shuffle and False: # this is deactivated and shuffling moved to data_gen
            sample_set = sample_set.shuffle(buffer_size=data_gen.n_elements)
        if is_training and False: # this is deactivated and repeating moved to data_gen to avoid OutOfRange Errors
            sample_set = sample_set.repeat()

        # create batch
        sample_set = sample_set.batch(batch_size)

        # prefetching at the end allows all previous ops to be scheduled independently of/parallel to any following ops
        if prefetch_n > 0:
            sample_set = sample_set.prefetch(prefetch_n)

        # create iterator
        [batch_img, batch_label, batch_weights] = sample_set.make_one_shot_iterator().get_next()

        # set shapes (dataset.batch() doesn't include batch_size information for shapes)
        batch_img.shape.dims[0] = tf.Dimension(batch_size)
        batch_label.shape.dims[0] = tf.Dimension(batch_size)
        batch_weights.shape.dims[0] = tf.Dimension(batch_size)

        logging.info("  | shapes: img %s | label %s | weights %s" %
                     (str(batch_img.shape), str(batch_label.shape), str(batch_weights.shape)))

        return batch_img, batch_label, batch_weights


# ##################################################################################
# ##############                       UTILS                            ############
# ##################################################################################

# HDF5 data generator contains logic on how to return samples from hdf5 files
class hdf5_generator:
    def __init__(self, file, keys=['data', 'label', 'weights'], augment=False,
                 repeat=False, cache_full_file=False, shuffle=False,
                 resample_n=None, name=None):
        """
        Create a HDF5 data generator. Is compatible to create a tf.dataset from it with
        sample_set = tf.data.Dataset.from_generator(
            data_gen,
            (tf.float32, tf.uint8, tf.float32),
            (tf.TensorShape(shape_img), tf.TensorShape(shape_label), tf.TensorShape(shape_weights))
        )

        :param file: dataset to read from
        :param keys: keys for img, label and weights in hdf5 dataset
        :param augment: allows to augment data already in the generator. needs to be implemented. can be slow
        :param repeat: allows to repeat the dataset in the generator, making it "infinetly large".
        tf.dataset.repeat() becomes obsolete when true.
        :param cache_full_file: read the complete hdf5 file before accessing single elements.
        :param shuffle: allows to pre-shuffle data in the generator.
        This is just random access to the elements,
        it doesn't ensure that all elements are seen during one epoch.
        :param resample_n: repeat a single element n times.
        This also works with shuffle (the shuffled element is repeated n times, then the next...)
        """
        self.file = file
        self.h5_keys = keys
        self.augment = augment
        self.repeat = repeat
        if not repeat and cache_full_file: self.repeat = True
        self.shuffle = shuffle
        self.cache_full_file = cache_full_file
        self.resample_n = resample_n
        self.name = name

        # parameters to adjust in code:
        self.debug = False  # for timing measurements (performance)
        self.element_axis = 0

    def __call__(self):
        if self.debug: start = timer()
        with h5py.File(self.file, 'r') as f:
            h5_keys = self.h5_keys  # for brevity in code

            # poke first dataset to get number of expected elements
            self.n_elements = f[h5_keys[0]].shape[self.element_axis]
            logging.info('init HDF5 [%s] data_gen | Resample: [%s] / Caching: [%s] / Repeat: [%s] / | Reading %s elements (poked \'%s\' with %s) from %s' %
                         ('' if self.name is None else self.name , str(self.resample_n), str(self.cache_full_file),
                          ('True' if self.repeat else str(self.repeat)),
                         str(self.n_elements), h5_keys[0], str(f[h5_keys[0]].shape), str(self.file)))

            if self.cache_full_file:
                # read data and swap axes (channels needs to be last dim) -> [elements, x, y, channels]
                img_eles = np.transpose(f[h5_keys[0]][:], [0, 2, 3, 1])
                label_eles = np.transpose(f[h5_keys[1]][:], [0, 2, 3, 1])
                weights_eles = np.transpose(f[h5_keys[2]][:], [0, 2, 3, 1])

            if self.debug: end = timer()
            if self.debug: logging.debug(" . setup in %.4f s, with caching: %s"
                                         % ((end - start), str(self.cache_full_file)))

            while True: # repeat data generator infinitely if repeat==True, otherwise run at least once (do-while-loop)
                for element_i in range(self.n_elements):
                    if self.shuffle:
                        # choose a random element
                        element_i = random.randrange(self.n_elements)

                    if self.resample_n is not None:
                        resample_i = 0 # init resample loop variable
                    while True: # resample if resample_n is given, otherwise run at least once (do-while-loop)
                        if self.debug: start = timer()
                        # TODO: Dynamic slicing (element_axis) doesn't work if cache_full_file==False
                        # f['data'][:].take(indices=element_i, axis=element_axis) is a way of replacing
                        # f['data'][ element_i, ...], allowing to dynamically pass an element axis as param for slicing
                        # however this has such a bad performance that dynamic element axis is disabled for now

                        # if not reading (already transposed) cached data, transpose to set n_channels as last dim
                        img = img_eles.take(indices=element_i, axis=self.element_axis) \
                            if self.cache_full_file else np.transpose(f[h5_keys[0]][element_i, ...], [1, 2, 0])
                        label = label_eles.take(indices=element_i, axis=self.element_axis) \
                            if self.cache_full_file else np.transpose(f[h5_keys[1]][element_i, ...], [1, 2, 0])
                        weights = weights_eles.take(indices=element_i, axis=self.element_axis) \
                            if self.cache_full_file else np.transpose(f[h5_keys[2]][element_i, ...], [1, 2, 0])

                        ######################################
                        # PYTHON PREPROCESS HOOK             #
                        # Python pre-processing              #

                        # this was moved to a tf_wrapper so it can be called for parallel threads
                        # img, label, weights = data_aug.augment(img, label, weights)

                        """
                        # This was needed when creating 4D-arrays [batch_size x y channels]: insert (dummy) batch_size dim
                        img = np.expand_dims(img, 0)
                        label = np.expand_dims(label, 0)
                        weights = np.expand_dims(weights, 0)
                        """
                        ######################################

                        if self.debug: end = timer()
                        if self.debug: logging.debug(" ... HDF5 data_gen element in %.4f s : img %s %s, label %s %s, weights %s %s"
                                                % ((end - start),
                                                   str(img.shape), str(img.dtype),
                                                   str(label.shape), str(label.dtype),
                                                   str(weights.shape), str(weights.dtype)))

                        yield img, label, weights

                        if self.resample_n is None: # break if not resampling
                            break
                        elif resample_i >= self.resample_n - 1: # break if resampled n times
                            break
                        else: resample_i += 1   # resample and count up
                if not self.repeat: break


def _preproc_tf_HDF5Data(imgs, label, weights):
    '''
    Helper class to ensure data read from HDF5 files is compatible with the net.
    (float32, shape [batch_size, x, y, n_channels)
    Necessary if data in HDF5 file is
    (float64, shape [batch_size, n_channels, x, y)

    :param imgs:
    :param label:
    :param weights:
    :return:
    '''

    # some preprocessing needed to make data compatible with net
    imgs = tf.cast(imgs, tf.float32) # tf generally works with float32
    label = tf.cast(label, tf.uint8) # so they can be converted to one-hot-encoding
    weights = tf.cast(weights, tf.float32)

    # tf equivalent of np.swapaxes (n_channels needs to be last dim)
    imgs = tf.transpose(imgs, [0, 2, 3, 1])
    label = tf.transpose(label, [0, 2, 3, 1])
    weights = tf.transpose(weights, [0, 2, 3, 1])

    return imgs, label, weights


def unify_ndims(arrays, ndims=None, axis=-1):
    '''
    Will attempt to squeeze or append empty (!) dimensions so that all tensors have the same ndims
    (i.e. number of dimensions of a Tensor, sometimes referred to as rank, order, degree).
    This will error if ndims is specified such that non-empty dimensions would need to be squeezed.

    :param arrays: an iterable / list of arrays.
    :param ndims: specifiy the desired ndims to unify to.
    If not specified will use the first element to determine ndims (ndims is returned).
    :param axis: By default -1, corresponds to squeezing/appending from the right. 0 means from the left.
    Any other axis n / -n will ignore the n first left / right dimensions and append to the left / right.
    :return: the unified tensors.
    Also returns ndims, as given to the function or as determined by the function if not specified
    '''
    arrays_u = []       # return list of tensors

    for array in arrays:
        # init on first tensor if no ndims is provided
        if ndims is None: ndims = len(array.shape)

        # squeeze dims [by default from the end (rightmost dim)]
        # this should error if the corresponding dim is not empty
        # (i.e. singular: E.g. [1024 1024 2] cannot be squeezed, [1024 1024 2 1] can be squeezed times)
        while(len(array.shape) > ndims):
            array = np.squeeze(array, axis=axis)

        # append dims [by default at the end (rightmost dim)]
        while (len(array.shape) < ndims):
            array = np.expand_dims(array, axis=axis)

        arrays_u.append(array)
    return arrays_u, ndims


# won't work!
def unify_tf_ndims(tensors, ndims=None, axis=-1):
    '''
    WATCH OUT! This won't work since tensors are not allowed to change shape in a conditional context :( (tf v1.6)

    Will attempt to squeeze or append empty (!) dimensions so that all tensors have the same ndims
    (i.e. number of dimensions of a Tensor, sometimes referred to as rank, order, degree).
    This will error if ndims is specified such that non-empty dimensions would need to be squeezed.

    :param tensors: an iterable / list of tensors.
    :param ndims: specifiy the desired ndims to unify to.
    If not specified will use the first element to determine ndims (ndims is returned).
    :param axis: By default -1, corresponds to squeezing/appending from the right. 0 means from the left.
    Any other axis n / -n will ignore the n first left / right dimensions and append to the left / right.
    :return: the unified tensors.
    Also returns ndims, as given to the function or as determined by the function if not specified
    '''
    tensors_u = []       # return list of tensors

    for tensor in tensors:
        # init on first tensor if no ndims is provided
        if ndims is None: ndims = tf.rank(tensor)

        # squeeze dims [by default from the end (rightmost dim)]
        # this should error if the corresponding dim is not empty
        # (i.e. singular: E.g. [1024 1024 2] cannot be squeezed, [1024 1024 2 1] can be squeezed times)
        if tensor.shape[-1] == 1:       # needs to be checked
            tensor = tf.while_loop(lambda tensor: tf.greater(tf.rank(tensor), ndims),
                                   lambda tensor: tf.squeeze(tensor, axis=axis),
                                   loop_vars = [tensor])

        # append dims [by default at the end (rightmost dim)]
        tensor = tf.while_loop(lambda tensor: tf.less(tf.rank(tensor), ndims),
                               lambda tensor: tf.expand_dims(tensor, axis=axis),
                               loop_vars = [tensor])

        tensors_u.append(tensor)

    return tensors_u, ndims






