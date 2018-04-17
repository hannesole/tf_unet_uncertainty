import os
import shutil
import logging
import time


def find_or_create_train_dir(name, output_dir, train_dir, continue_training=False):
    # create train directory train_dir if it doesn't exist
    if train_dir is None:  # use default naming if no parameter is passed for train_dir
        # create train_dir with name (by default current time)
        train_dir = os.path.join(output_dir, name)

        if os.path.exists(train_dir):
            if continue_training:
                logging.info('Reusing existing directory %s ' % train_dir)
                return train_dir, True
            else:
                if name == 'overwrite':
                    # remove before recreating
                    logging.info('Removing existing directory %s ' % train_dir)
                    shutil.rmtree(train_dir)
                else:
                    # avoid overwriting (by appendin _X to name if already exists)
                    train_dir = train_dir + '_'
                    num = 2
                    while os.path.exists(train_dir + str(num)):
                        num = num + 1
                    train_dir = train_dir + str(num)

        logging.info('Created train_dir for writing model and other output: %s ' % train_dir)
        os.mkdir(train_dir)
        return train_dir, False
    else:
        # make sure train_dir exists
        if not os.path.exists(train_dir):
            logging.info('Created train_dir for writing model and other output: %s ' % train_dir)
            os.mkdir(train_dir)
            return train_dir, False
        else:
            logging.info('Reusing train_dir: %s ' % train_dir)
            return train_dir, continue_training


def find_or_create_test_dir(test_dir, train_dir):
    # create test_dir if it doesn't exist. Use either default naming or a given parameter.
    if test_dir is None:
        # create test_dir with current time if none is specified
        test_dir = train_dir + os.sep + "prediction" + '_' + time.strftime("%Y-%m-%d_%H%M")
        # avoid overwriting (by appendin _X to name if already exists)
        if os.path.exists(test_dir):
            test_dir = test_dir + '_'
            num = 2
            while os.path.exists(test_dir + str(num)):
                num = num + 1
            test_dir = test_dir + str(num)
        os.mkdir(test_dir) # create later
        logging.info('Create test_dir %s (name avoids overwriting).' % test_dir)
        return test_dir
    else:
        # make sure test_dir exists
        if not os.path.exists(test_dir):
            os.mkdir(test_dir) # create later
            logging.info('Create test_dir %s ' % test_dir)
        else:
            logging.info('Using existing test_dir %s ' % test_dir)
        return test_dir


def find_or_create_code_copy_dir(file_src, code_copy_dir, train_dir):
    if code_copy_dir is not None:
        if code_copy_dir == 'train_dir':
            # copy script to train_dir
            code_copy_dir = train_dir + os.sep + 'py_src'
        if not os.path.exists(code_copy_dir):
            logging.info('Created py source dir: ' + code_copy_dir)
            os.mkdir(code_copy_dir)

        copy_file_path = code_copy_dir + os.sep + os.path.basename(file_src)
        copied = copy_file(__file__, copy_file_path)
        logging.info('Copied py source to: ' + copied)
        return code_copy_dir
    else:
        return None


def find_or_create_config_path(base_dir, config_name = 'config.ini', config_template = 'config_template.ini'):
    """ This checks first for a config.ini in base_dir. If none is found, config_template is copied. """
    config_path = base_dir + os.sep + config_name

    if os.path.exists(config_path):
        logging.info('Using config found at %s' % (config_path))
        return config_path
    else:
        logging.info('Found no config at %s' % (config_path))
        shutil.copy(config_template, config_path)
        logging.info('Copied %s to create config at %s' % (config_template, config_path))
        return config_path


def copy_file(file_src, file_dest, overwrite=False):
    """ Copies a file from file_src to file_dest and appends time if file_dest exists and overwrite=False """
    if os.path.exists(file_dest) and not overwrite:
        file_dest = file_dest + '_' + time.strftime("%Y-%m-%d_%H%M")
    shutil.copy(file_src, file_dest)
    return file_dest