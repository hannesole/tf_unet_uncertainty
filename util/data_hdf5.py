# HDF5 DATA LIB
# =============
# Functions to build and modify hdf5 files. Mainly to merge split hdf5 files into one,
# either using a table structure or maintaining the dset structure.
#
# Author: Hannes Horneber
# Date: 2018-03-18

import tensorflow as tf
import h5py
import numpy as np
import logging
import os
import glob
import matplotlib.pyplot as plt
from unet import data_layers
import tables
import time

def _find_hdf5_files(hdf5_folder, filter_suffix=None):
    '''
        expects a folder containing .h5 files
        ALL .h5 files in this folder will be merged, unless you provide a filter suffix.

    :param hdf5_folder: folder containing .h5 files
    :param filter_suffix: return only files containing filter_suffix in their filename
    :return: .h5 files in folder
    '''
    h5_files = glob.glob(hdf5_folder+ os.sep +'*.h5')
    if filter_suffix is None:
        return h5_files
    else:
        return [name for name in h5_files if filter_suffix in name]


def _inspect_hdf5_file(hdf5_file, on_dset=False):
    '''
    Print and return basic info about datasets in hdf5_file.

    :param hdf5_file:
    :return: keys and dsets of a file
    '''
    f = h5py.File(hdf5_file, 'r')
    keys = list(f.keys())
    logging.debug('HDF5inspect: %s keys in file %s ' % (str(keys), str(hdf5_file)))

    if on_dset:
        dsets = []
        for key in keys:
            dset = f[key]
            logging.debug(' > %s: shape %s, dtype %s' %(key, str(dset.shape), str(dset.dtype)))
            dsets.append(dset)
        return keys, dsets
    else: return keys


def _validate_datasources(datasources, element_axis=None, on_dsets=False):
    '''
    Asserts that hdf5 files (data sources) have same keys and if on_dsets is True, also checks shapes and dtypes.
    This is by default False to put off reading the actual dsets as much as possible
    (since they have to be read during filling the table anyways).

    :param datasources: list of hdf5 files
    :param element_axis: (optional) if given, additional checks could be made. Not implemented yet. By default None.
    :param on_dsets: (optional) by default False. If true, dsets are checked (not only keys)
    :return: joinable_keys and validated_dsets if on_dsets.
    '''
    #TODO: automatically detect element_axis (the only axis that allows different values for different files in dataset.shape)
    # then return: element_axis over which all files are joined.

    previous_keys = None    # init to check first iteration of loop
    for file in datasources:
        if previous_keys is None:
            # initialize on first file
            if on_dsets: previous_keys, previous_dsets = _inspect_hdf5_file(file, on_dset=on_dsets)
            else: previous_keys = _inspect_hdf5_file(file)

            # TODO: only return joinable keys (insted of assertions with errors, remove erroneous keys)
            #  currently simply all keys of first file are returned
            joinable_keys = previous_keys
            if on_dsets: validated_dsets = previous_dsets
        else:
            if on_dsets: keys, dsets = _inspect_hdf5_file(file, on_dset=on_dsets)
            else: keys = _inspect_hdf5_file(file)

            assert len(keys) == len(previous_keys), \
                "HDF5 files have different amount of keys (%s and %s)" %(str(previous_keys), str(keys))
            for i in range(len(keys)):
                assert keys[i] == previous_keys[i], \
                    "HDF5 files have different keys (%s and %s)" %(str(previous_keys), str(keys))

                # some checks can only be performed if dsets are read
                if on_dsets:
                    assert dsets[i].dtype == previous_dsets[i].dtype, \
                        "HDF5 datasets have different dataformats (%s and %s)" %(str(previous_dsets[i].dtype), str(dsets[i].dtype))
                    assert len(dsets[i].shape) == len(previous_dsets[i].shape), \
                        "HDF5 arrays have different n_dim (%s and %s)" % (str(previous_dsets[i].shape), str(dsets[i].shape))

                    if not np.array_equal(dsets[i].shape, previous_dsets[i].shape):
                        logging.info("Warning: HDF5 arrays have same n_dim but different shapes (%s and %s)" % (str(previous_dsets[i].shape), str(dsets[i].shape)))

            # set variables for next iteration
            previous_keys = keys
            if on_dsets: previous_dsets = dsets

    if on_dsets: return joinable_keys, validated_dsets
    else: return joinable_keys


def _get_dtypes(datasources, element_axis, keys=None, get_size_of_complete_data=False):
    '''
    Returns the datatypes of datasets stored in datasources.
    Without the get_size_of_complete_data flag set to True, this only reads the first file to determine dtype and shapes.
    If no keys are provided, this assumes keys are already validated and uses the first file to determine
    the list of keys.

    :param datasources: list of source files
    :param element_axis: axis with which elements are accessed (e.g. is 0 for [n_elements 1024 1024 3]). Usually 0.
    :param keys: (optional) will be generated from first file if not provided
    :param get_size_of_complete_data: by default False, if True will return shapes_complete,
    a list of shapes where each shape represents the size of a dset reconstructed from all datasource files
    :return: keys, dtypes (list of dtypes per dset), elements_shapes (list of element_shape per dset).
    If get_size_of_complete_data also returns shapes_complete
    '''
    first = True    # flag for first iteration
    for hdf5_file in datasources:
        with h5py.File(hdf5_file, 'r') as f:
            # init shapes, get dtypes only from first file
            if first:
                # fetch and use keys of first dset if no keys are provided
                if keys is None: keys = list(f.keys())
                # get dtype and shapes
                dtypes = [f[key].dtype for key in keys]
                elements_shapes = [list(f[key].shape) for key in keys]
                for shape in elements_shapes:
                    del shape[element_axis]

                # if size_of_complete_data is not needed, reading the first file and dset is enough
                if not get_size_of_complete_data: return keys, dtypes, elements_shapes

                shapes_complete = [list(f[key].shape) for key in keys]
                first = False   # remove first flag
            else:
                # update total number of elements in shapes for each dataset
                for i in range(len(keys)):
                    # add number of elements from file f to shape
                    shapes_complete[i][element_axis] += f[keys[i]].shape[element_axis]

    return keys, dtypes, elements_shapes, shapes_complete


def _fill_table(datasources, val_keys, element_axis, table_row, cache_full_file=False, row_wise=False):
    '''
    Writes elements from datasets that are extracted from datasource files into a table (is referenced by table_row).
    By default writes column-wise with putting off reading the whole file for as long as possible.
    If that doesn't matter you may use cache_full_file.

    :param datasources:
    :param val_keys:
    :param element_axis:
    :param table_row:
    :param cache_full_file:
    :param row_wise:
    :return:
    '''
    logging.info(' >> filling table with rows %s' % str(table_row))
    for hdf5_file in datasources:
        with h5py.File(hdf5_file, 'r') as f:
            logging.info('    >> reading file: %s' % hdf5_file)

            # poke first dataset to get number of expected elements
            n_elements = f[val_keys[0]].shape[element_axis]
            logging.info('    >> adding %s elements/rows (poked %s with %s)' % (str(n_elements), val_keys[0], str(f[val_keys[0]].shape)))

            if cache_full_file:
                dset_content = [f[key][:] for key in val_keys]
                logging.info(' >> cached content from file: %s' % (str(dset_content[0].shape)))

            if row_wise:
                # fill row-wise
                for element_i in range(n_elements):
                    for key_i in range(len(val_keys)):
                        table_row[val_keys[key_i]] = (
                            # use cached dset_content if flag is true, otherwise use h5py file access
                            dset_content[val_keys[key_i]] if cache_full_file else f[val_keys[key_i]][:]
                        ).take(indices=element_i, axis=element_axis)
                        table_row.append()
            else:
                # fill column-wise (for loops swapped)
                for key_i in range(len(val_keys)):
                    for element_i in range(n_elements):
                        table_row[val_keys[key_i]] = (
                            # use cached dset_content if flag is true, otherwise use h5py file access
                            dset_content[val_keys[key_i]] if cache_full_file else f[val_keys[key_i]][:]
                        ).take(indices=element_i, axis=element_axis)
                        table_row.append()



def merge_hdf5(hdf5_folder, dest_folder=None, dest_filename=None, datasources=None, element_axis=0, validate=False,
               build_table=False, chunks=False):
    '''
    Merges HDF5 files into one file.
    Assumes that all files contain the same datasets with the same shapes and dtypes, if assumption is not safe,
    set validate flag to True (False by default).
    By default the first axis is the "element_axis", meaning that it represents the index axis over which all others are
    joined (e.g.: [5, 3, 256, 256] and [4, 3, 256, 256] will be joined into [9, 3, 256, 256], whereas unequal numbers in
    other dimensions will result in errors.

    :param hdf5_folder: folder containing hdf5 files to be merged.
    If the datasources parameter is explicitly given (AND dest_folder is not set) this folder will be used as dest_folder.
    If both are set, this parameter is actually obsolete (still not optional for safety though).
    :param dest_folder: (optional) folder to store the resulting merged HDF5 file. By default this is the same as hdf5folder.
    :param dest_filename: (optional) name of the resulting file. By default 'merged.h5'
    :param datasources: (optional) list of hdf5 files to be merged. By default this will be generated from files in hdf5folder.
    :param element_axis: (optional) shape dimension (axis) that represents elements. By default 0 (the first dim).
    :param build_table: (optional) merges data into a table ('data_table') with keys as columns. By default False.
    :param chunks: (optional) writes dset in chunked mode. Chunksize is guessed. By default False.
    :return: Returns string with full path to created file
    '''
    if dest_filename is None:
        dest_filename = 'merged_%s.h5' % ('table' if build_table else 'dset')
    if dest_folder is None:
        dest_folder = hdf5_folder
    if datasources is None:
        datasources = _find_hdf5_files(hdf5_folder)

    # generate save path for resulting file
    dest_filename = dest_folder + os.sep + dest_filename

    logging.info('build_hdf5_file >> merging %s files: \n %s' % (str(len(datasources)), str(datasources)))
    logging.info(' >> saving to: ' + dest_filename)

    # validation is optional
    if validate: val_keys, val_dsets, element_axis = _validate_datasources(datasources, element_axis=element_axis)

    # get key, dtype and shapes of elements per dataset from the datasource files
    val_keys, dtypes, element_shapes = _get_dtypes(datasources, element_axis=element_axis,
                                                   keys=val_keys if 'myVar' in locals() else None)

    logging.info('  >> keys %s: ' % str(val_keys))
    logging.info('  >> dtypes %s: ' % str(dtypes))
    logging.info('  >> element shapes %s: ' % str(element_shapes))

    if build_table:
        # define how table looks like
        description = {val_keys[i]: tables.Col.from_type(str(dtypes[i]), shape=element_shapes[i]) for i in range(len(val_keys))}

        # create a file, a group-node and attach a table to i (can show structure with print(h5file))
        if not os.path.exists(dest_folder): os.mkdir(dest_folder)   # make sure folder exists
        h5file = tables.open_file(dest_filename, mode="w", title="Merged Dataset with %s" % (str(val_keys)))
        table = h5file.create_table("/", 'data_table', description, "Collected data with %s" % (str(val_keys)))
        # create row object to fill table
        sample = table.row

        _fill_table(datasources, val_keys, element_axis, sample)
        logging.info(' >> finished writing, flushing IO buffer')
        table.flush()
        h5file.close()
        logging.info(' >> done :)')
    else:
        #simply merge datasets
        for key in val_keys:
            first = True
            # read and concatenate data from all files
            for hdf5_file in datasources:
                f = h5py.File(hdf5_file, 'r')
                if first:
                    full_dset = f[key][:]
                    logging.info('    >> init dset [%s]: %s' % (key, str(full_dset.shape)))
                    first = False
                else:
                    full_dset = np.concatenate((full_dset, f[key][:]), axis=element_axis)
                    logging.info('    >> growing dset [%s]: %s' % (key, str(full_dset.shape)))
                f.close() # all reading done, go to next file

            # write concatenated (merged) dset to output file
            fo = h5py.File(dest_filename, 'a')
            logging.info('    >> writing dset [%s] to: %s' % (key, dest_filename))
            fo.create_dataset(key, data=full_dset, chunks=chunks)
            fo.close()

    return dest_filename



    # for hdf5_file in datasources:
    #     with h5py.File(hdf5_file, 'r') as f:
    #         logging.info('    >> reading file: %s' % hdf5_file)
    #
    #         # poke first dataset to get number of expected elements
    #         n_elements = f[val_keys[0]].shape[element_axis]
    #         logging.info('    >> adding %s elements/rows (poked %s with %s)' % (str(n_elements), val_keys[0], str(f[val_keys[0]].shape)))
    #
    #
    #         # fill column-wise (for loops swapped)
    #         for key_i in range(len(val_keys)):
    #             for element_i in range(n_elements):
    #                 my_data = (f[val_keys[key_i]][:]).take(indices=element_i, axis=element_axis)
    #                 sample[val_keys[key_i]] = my_data
    #                 sample.append




def merge_hdf5_table(hdf5_folder, dest_folder=None, dest_filename=None, datasources=None, element_axis=0, validate=False):
    '''
    Merges HDF5 files into one file.
    Assumes that all files contain the same datasets with the same shapes and dtypes, if assumption is not safe,
    set validate flag to True (False by default).
    By default the first axis is the "element_axis", meaning that it represents the index axis over which all others are
    joined (e.g.: [5, 3, 256, 256] and [4, 3, 256, 256] will be joined into [9, 3, 256, 256], whereas unequal numbers in
    other dimensions will result in errors.

    :param hdf5_folder: folder containing hdf5 files to be merged.
    If the datasources parameter is explicitly given (AND dest_folder is not set) this folder will be used as dest_folder.
    If both are set, this parameter is actually obsolete (still not optional for safety though).
    :param dest_folder: (optional) folder to store the resulting merged HDF5 file. By default this is the same as hdf5folder.
    :param dest_filename: (optional) name of the resulting file. By default 'merged.h5'
    :param datasources: (optional) list of hdf5 files to be merged. By default this will be generated from files in hdf5folder.
    :param element_axis: (optional) shape dimension (axis) that represents elements. By default 0 (the first dim).
    :return:
    '''
    if dest_filename is None:
        dest_filename = 'merged.h5'
    if dest_folder is None:
        dest_folder = hdf5_folder
    if datasources is None:
        datasources = _find_hdf5_files(hdf5_folder)

    # generate save path for resulting file
    dest_filename = dest_folder + os.sep + dest_filename

    logging.info('build_hdf5_file >> merging %s files: \n %s' % (str(len(datasources)), str(datasources)))
    logging.info(' >> saving to: ' + dest_filename)

    # validation should be optional
    if validate: val_keys, val_dsets, element_axis = _validate_datasources(datasources, element_axis=element_axis)

    # get key, dtype and shapes of elements per dataset from the datasource files
    val_keys, dtypes, element_shapes = _get_dtypes(datasources, element_axis=element_axis)

    logging.info('  >> keys %s: ' % str(val_keys))
    logging.info('  >> dtypes %s: ' % str(dtypes))
    logging.info('  >> element shapes %s: ' % str(element_shapes))

    # define how table looks like
    description = {val_keys[i]: tables.Col.from_type(str(dtypes[i]), shape=element_shapes[i]) for i in range(len(val_keys))}

    # create a file, a group-node and attach a table to i (can show structure with print(h5file))
    if not os.path.exists(dest_folder): os.mkdir(dest_folder)   # make sure folder exists
    h5file = tables.open_file(dest_filename, mode="w", title="Merged Dataset with %s" % (str(val_keys)))
    table = h5file.create_table("/", 'data_table', description, "Collected data with %s" % (str(val_keys)))
    # create row object to fill table
    sample = table.row

    _fill_table(datasources, val_keys, element_axis, sample)
    # for hdf5_file in datasources:
    #     with h5py.File(hdf5_file, 'r') as f:
    #         logging.info('    >> reading file: %s' % hdf5_file)
    #
    #         # poke first dataset to get number of expected elements
    #         n_elements = f[val_keys[0]].shape[element_axis]
    #         logging.info('    >> adding %s elements/rows (poked %s with %s)' % (str(n_elements), val_keys[0], str(f[val_keys[0]].shape)))
    #
    #
    #         # fill column-wise (for loops swapped)
    #         for key_i in range(len(val_keys)):
    #             for element_i in range(n_elements):
    #                 my_data = (f[val_keys[key_i]][:]).take(indices=element_i, axis=element_axis)
    #                 sample[val_keys[key_i]] = my_data
    #                 sample.append

    logging.info(' >> finished writing, flushing IO buffer')
    table.flush()
    h5file.close()
    logging.info(' >> done :)')


def test_hdf5(data_path=None, output_dir=None, shape_img=[1024, 1024, 3], shape_label=[1024, 1024, 1],
                    batch_size=1, shuffle=True, n_samples=10, table_access=False ):
    '''
    Tests a HDF5 data layer by running a tf session to generate some batches and write to a sample folder (output_dir).

    :param data_path:
    :param mode:
    :param shape_img:
    :param shape_label:
    :param batch_size:
    :param shuffle:
    :param output_dir:
    :param n_samples:
    :param table_access: False (by default) if the HDF5 dataset style is used, True if HDF5 table style is used.
    This determines which layer is used.
    :return:
    '''
    if table_access:
        if data_path is None: data_path = 'data/hdf5/std_data_v0_2_pdf/train/merged/train.h5'
        if output_dir is None: output_dir = 'data/hdf5/std_data_v0_2_pdf/train/merged/samples_' + time.strftime("%Y-%m-%d_%H%M")
        logging.debug("test HDFLayer >> %s | batch_size %s" % (data_path, str(batch_size)))
        loader, reader, img_tensor, label_tensor, weights_tensor = data_layers.data_HDF5TableQueue(data_path, batch_size=batch_size)
    else:
        if data_path is None: data_path = 'data/hdf5/std_data_v0_2_pdf/train/std_data_v0_2_pdf_train_1.h5'
        if output_dir is None: output_dir = 'data/hdf5/std_data_v0_2_pdf/train/samples_' + time.strftime("%Y-%m-%d_%H%M")
        logging.debug("test HDFTableLayer >> %s | batch_size %s" % (data_path, str(batch_size)))
        loader, reader, img_tensor, label_tensor, weights_tensor = data_layers.data_HDF5Queue(data_path, batch_size=batch_size)
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    with tf.Session() as sess:
        with loader.begin(sess):
            for batch_n in range(n_samples):
                img_batch, label_batch, weights_batch = sess.run([img_tensor, label_tensor, weights_tensor])
                # [batch_size, n_channels, x, y] -> [batch_size, x, y, n_channels]
                # img_batch = np.swapaxes(img_batch, 1, 3)
                # label_batch = np.swapaxes(label_batch, 1, 3)
                # weights_batch = np.swapaxes(weights_batch, 1, 3)

                logging.debug(" img_batch loop: %s %s | label_batch %s %s" %
                              (str(img_batch.shape), str(img_batch.dtype), str(label_batch.shape),
                               str(label_batch.dtype)))
                # logging.debug(" label_batch %s" % (str(label_batch)))
                img_batch = img_batch * 255  # rgb values
                img_batch = img_batch.astype(np.uint8)
                label_batch = label_batch.astype(np.uint8)
                # weights_batch = weights_batch * 255  # rgb values
                # weights_batch = weights_batch.astype(np.uint8)

                for j in range(batch_size):
                    logging.debug("   img: %s/%s | shape: %s | %s"
                                  % (str(j), str(batch_size), str(img_batch[j, :, :, 0:3].shape),
                                     str(label_batch[j, ...].shape)))
                    my_img = img_batch[j, :, :, 0:3]

                    plt.subplot(3, batch_size, (j * 3) + 1)
                    plt.imshow(my_img)
                    plt.subplot(3, batch_size, (j * 3) + 2)
                    plt.imshow(np.squeeze(label_batch[j, ...]))
                    plt.subplot(3, batch_size, (j * 3) + 3)
                    plt.imshow(np.squeeze(weights_batch[j, ...]), cmap='jet')
                    if batch_n == 0 and j == 0: plt.colorbar()

                logging.info(
                    ' read_tf_records >> writing batch to: ' + (output_dir + os.sep + 'tile_' + str(batch_n) + '.png'))
                plt.savefig(output_dir + os.sep + 'batch_' + str(batch_n) + '.png')
                # plt.close(fig)

    reader.close()
    sess.close()