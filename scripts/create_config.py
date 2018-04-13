# CREATE CONFIG FILE
# =========================
#
# This script creates a config file (template).
#
# Author: Hannes Horneber
# Date: 2018-04-10


import configparser
import sys
from collections import OrderedDict
#config = configparser.ConfigParser(dict_type=OrderedDict)
config = configparser.ConfigParser()

# DEFAULT values apply to training and testing if not specified otherwise in the respective section
config['DEFAULT'] = OrderedDict([
    ('copy_script_dir' , 'train_dir'),

    ('shuffle', 'True'),      # shuffle in input
    ('augment', 'True'),      # randomly augment data input
    ('resample_n', 'None'),   # resample same image to use always the same image in training
    ('batch_size', '1'),      # simultaneously process [batch_size] images (GPU mem is limit)

    # working configurations for 12 GB GPU RAM:
    # size 512x512: batch_size= 1 | 2 | 8 ; n_start_features= 64 | 64 | 16
    ('n_contracting_blocks' , '5'),
    ('n_start_features' , '64'),
    ('resize' , '[512, 512]'),    # None # json.loads(config.get("DEFAULT","resize"))
    ('resize_method' , 'None'),   # "scale" # None # "scale", "center_crop" or "random_crop" (default)

    ('norm_fn' , 'None'),  # tc.layers.batch_norm  #
    ('normalizer_params' , 'None'),

    ('data_layer_type' , 'hdf5'),     # see unet model / data_layers for specifics
    ('shape_img' , '[1024, 1024, 4]'),    # of a single image
    ('shape_label' , '[1024, 1024, 1]'),  # of a single label mask
    ('shape_weights' , '[1024, 1024, 1]') # of a single weight mask
])

# normalizer params is a dict in itself. Might be circumstancial to expose this as a config option (needs to be parsed)
config['normalizer_params'] = OrderedDict([
    ('is_training', 'None'),                  # set to None if it should correspond to Phase
    ('decay', '0.9'),                         # more stable than default (0.999)
    ('zero_debias_moving_mean', 'True')       # for stability
])

config['TRAIN'] = OrderedDict([
    ('train_name' , 'augment_saver'),
    ('init_learning_rate' , '0.0001'),    # initial learning rate
    ('max_iter' , '60000'),   # maximum iterations
    ('keep_prob' , '0.9'),    # dropout - 1
    ('optimizer' , 'Adam'),

    ('prefetch_n' , '32'),
    ('prefetch_threads' , '16'),

    ('saver_interval' , '1000')
])

config['TEST'] = OrderedDict([
    ('n_samples' , str(20)),      # ... of n_samples
    ('resample_n' , str(20)),     # resample same image to calculate uncertainty, N or None
    ('resize_method' , "scale"),  # "center_crop"
    ('keep_prob' ,  '1.0')        # use 1.0 for no dropout during test-time
])

# Optionally order alphabetically:

# #Order the content of defaults alphabetically
# config._defaults = OrderedDict(sorted(config._defaults.items(), key=lambda t: t[0]))
#
# #Order the content of each section alphabetically
# for section in config._sections:
#     config._sections[section] = OrderedDict(sorted(config._sections[section].items(), key=lambda t: t[0]))
#
# # Order all sections alphabetically
# config._sections = OrderedDict(sorted(config._sections.items(), key=lambda t: t[0] ))

config.write(sys.stdout)

# use this to write to file
# with open('config_generated.ini', 'w') as configfile:
#     config.write(configfile)