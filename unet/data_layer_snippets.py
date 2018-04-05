# def _load_file(self, path, dtype=np.float32):
#   return np.array(Image.open(path), dtype)
#   # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))
#
# image_name = data_files[file_idx]
# img = _load_file(image_name, np.float32)
#
# # test for number of channels
# channels = 1 if len(img.shape) == 2 else img.shape[-1]
#
# label_name = image_name.replace(data_suffix, mask_suffix)
# label = _load_file(label_name, np.bool)
# return img, label
#
#
# import tensorflow as tf
# import cv2
#
# # Use a custom OpenCV function to read the image, instead of the standard
# # TensorFlow `tf.read_file()` operation.
# def _read_py_function(filename, label):
#     np.array(Image.open(path), dtype)
#     image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     return image_decoded, label
#
# # Use standard TensorFlow operations to resize the image to a fixed shape.
# def _resize_function(image_decoded, label):
#     image_decoded.set_shape([None, None, None])
#     image_resized = tf.image.resize_images(image_decoded, [28, 28])
#     return image_resized, label
#
# filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
# labels = [0, 37, 29, 1, ...]
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(
#     lambda filename, label: tuple(tf.py_func(
#         _read_py_function, [filename, label], [tf.uint8, label.dtype])))
# dataset = dataset.map(_resize_function)
#
#
# # Reads an image from a file, decodes it into a dense tensor, and resizes it
# # to a fixed shape.
# def _parse_function(filename, label):
#   image_string = tf.read_file(filename)
#   image_decoded = tf.image.decode_image(image_string)
#   image_resized = tf.image.resize_images(image_decoded, [28, 28])
#   return image_resized, label
#
# # A vector of filenames.
# filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
#
# # `labels[i]` is the label for the image in `filenames[i].
# labels = tf.constant([0, 37, ...])
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(_parse_function)
#
#
# """
#
# import cv2
#
# # Use a custom OpenCV function to read the image, instead of the standard
# # TensorFlow `tf.read_file()` operation.
# def _read_py_function(filename, label):
#   image_decoded = cv2.imread(image_string, cv2.IMREAD_GRAYSCALE)
#   return image_decoded, label
#
# # Use standard TensorFlow operations to resize the image to a fixed shape.
# def _resize_function(image_decoded, label):
#   image_decoded.set_shape([None, None, None])
#   image_resized = tf.image.resize_images(image_decoded, [28, 28])
#   return image_resized, label
#
# filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
# labels = [0, 37, 29, 1, ...]
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(
#     lambda filename, label: tuple(tf.py_func(
#         _read_py_function, [filename, label], [tf.uint8, label.dtype])))
# dataset = dataset.map(_resize_function)
#
#
# """
#
#
#
# """
# # Load the training data into two NumPy arrays, for example using `np.load()`.
# with np.load("/var/data/training_data.npy") as data:
#   features = data["features"]
#   labels = data["labels"]
#
# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]
#
# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
#
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# # [Other transformations on `dataset`...]
# dataset = ...
# iterator = dataset.make_initializable_iterator()
#
# sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                           labels_placeholder: labels})
# """
#
#
# # Reads an image from a file, decodes it into a dense tensor, and resizes it
# # to a fixed shape.
# def _parse_function(filename, label):
#   image_string = tf.read_file(filename)
#   image_decoded = tf.image.decode_image(image_string)
#   image_resized = tf.image.resize_images(image_decoded, [28, 28])
#   return image_resized, label
#
# # A vector of filenames.
# filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
#
# # `labels[i]` is the label for the image in `filenames[i].
# labels = tf.constant([0, 37, ...])
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(_parse_function)
#
#
# """
# class BaseDataProvider(object):
#     """
#     Abstract base class for DataProvider implementation. Subclasses have to
#     overwrite the `_next_data` method that load the next data and label array.
#     This implementation automatically clips the data with the given min/max and
#     normalizes the values to (0,1]. To change this behavoir the `_process_data`
#     method can be overwritten. To enable some post processing such as data
#     augmentation the `_post_process` method can be overwritten.
#
#     :param a_min: (optional) min value used for clipping
#     :param a_max: (optional) max value used for clipping
#     """
#
#     channels = 1
#     n_class = 2
#
#     def __init__(self, a_min=None, a_max=None):
#         self.a_min = a_min if a_min is not None else -np.inf
#         self.a_max = a_max if a_min is not None else np.inf
#
#     def _load_data_and_label(self, mask=True):
#         if mask:
#             data, label = self._next_data(mask=mask)
#         else: data = self._next_data(mask=mask)
#
#         train_data = self._process_data(data)
#         if mask:
#             labels = self._process_labels(label)
#             # post_processing hook
#             train_data, labels = self._post_process(train_data, labels)
#
#         nx = data.shape[1]
#         ny = data.shape[0]
#
#         if mask:
#             return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)
#         return train_data.reshape(1, ny, nx, self.channels)
#
#     def _process_labels(self, label):
#         if self.n_class == 2:
#             nx = label.shape[1]
#             ny = label.shape[0]
#             labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
#             labels[..., 1] = label
#             labels[..., 0] = ~label
#             return labels
#
#         return label
#
#     def _process_data(self, data):
#         # norm_fn
#         data = np.clip(np.fabs(data), self.a_min, self.a_max)
#         data -= np.amin(data)
#         data /= np.amax(data)
#         return data
#
#     def _post_process(self, data, labels):
#         """
#         Post processing hook that can be used for data augmentation
#
#         :param data: the data array
#         :param labels: the label array
#         """
#         return data, labels
#
#     def __call__(self, n, mask=True):
#         """
#         A call of the class generates and returns n images.
#
#         :param n: the number of data files that are generated
#         :param mask: (optional) set to False if no mask file is provided/needed
#         """
#         if mask:
#             train_data, labels = self._load_data_and_label(mask=mask)
#         else:
#             train_data = self._load_data_and_label(mask=mask)
#
#         nx = train_data.shape[1]
#         ny = train_data.shape[2]
#
#         X = np.zeros((n, nx, ny, self.channels))
#         Y = np.zeros((n, nx, ny, self.n_class))
#
#         X[0] = train_data
#         if mask:
#             Y[0] = labels
#         for i in range(1, n):
#             if mask:
#                 train_data, labels = self._load_data_and_label(mask=mask)
#             else:
#                 train_data = self._load_data_and_label(mask=mask)
#
#             X[i] = train_data
#             if mask:
#                 Y[i] = labels
#
#         if mask:
#             return X, Y
#         return X
# """