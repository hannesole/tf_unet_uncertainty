# CONFIGURATION UTILS
# =========================
#
# Functions/classes to help handling configuration.
#
# Author: Hannes Horneber
# Date: 2018-04-10

from json import loads as parse_list
from collections import OrderedDict
import configparser
import logging

def opts_to_str(opts, title='SCRIPT', width = 96):
    '''Prints the opts class as box with all settings'''
    # vars(opts) == opts.__dict__  # but in different order!
    my_opts = opts.get_attr_val_list()
    opts_str = '_' * width + ('\n| %s OPTIONS:') % title  + ' ' * (width - 11 - len(title) - 1) + '|'
    for opt in my_opts:
        opts_str = opts_str + '\n|   ' + opt[0] + '.' * (width // 2 - len(opt[0]) - 5) + ':  ' \
                   + str(opt[1]).replace('\n', ' ') + ' ' * (width // 2 - len(str(opt[1])) - 3) + '|'
    opts_str = opts_str + '\n' + '_' * width + '\n'
    return opts_str


def parse_implicitly(s):
    """ This parses strings to Python types (int, float, bool, NoneType, list) """
    try:  # try for float second
        try:  # try for int first
            return int(s)
        except ValueError:  # not an int
            return float(s)
    except ValueError:  # not a float
        # process string
        if s.lower() == 'true': return True
        if s.lower() == 'false': return False
        if s.lower() == 'none': return None
        if s.startswith('['): return parse_list(s)
        return s  # fallback on string


def parse_implicitly_extended(config_prox, attr):
    """" This returns a class for norm_fn and a dict of options for norm_fn_params """

    # Wrapping in a try-catch phrase allows referencing options that are neither defined
    # in config.ini nor the class by returning None for those.
    try:
        attr_val = parse_implicitly(config_prox[attr])
    except KeyError:
        logging.warning('Key %s not in config! Defaults to None.' % attr)
        return None

    if attr == 'norm_fn':
        # get actual class from string
        if attr_val is None or attr_val is False:
            return None
        elif attr_val == 'tc.layers.batch_norm':
            from tensorflow import contrib as tc
            return tc.layers.batch_norm
        else:
            return None
    elif attr == 'norm_fn_params':
        # get actual class from string
        if attr_val is None or attr_val is False:
            # False or None corresponds to "don't load" (if norm_fn = None/False)
            # or "load defaults" (if norm_fn is specified)
            return None
        elif attr_val is True:
            # create dict from section if norm_fn_params_val is True
            return dict(
                [(norm_fn_attr.replace('norm_fn_param_', ''), parse_implicitly(config_prox[norm_fn_attr]))
                 for norm_fn_attr in OrderedDict(config_prox.items())
                 if norm_fn_attr.startswith('norm_fn_param_')])
        else:
            return None
    else:
        return attr_val


class config_decorator:
    """
    Allows to read from a configparser object in a struct style.
    i.e. with config.key instead of config_proxy['key']
    This automatically parses the datatype of strings returned
    from the config object.

    Usage:
    config = configparser.ConfigParser()
    config.read('config.ini')
    opts = config_decorator(config['section'])

    attr1 = opts.key1  # equivalent to config['section']['key1']
    """
    def __init__(self, config_prox):
        self.config_prox = config_prox

    def __getattr__ (self, attr):
        """
        Overrides default attribute behavior:
        Will read an attribute that isn't found within the class
        from the config_proxy (defined when __init__())
        The read value is implicitly parsed to a Python type.

        :param attr: key of a config value
        :return: value of attr in config_proxy with parsed data type
        """
        # Wrapping in a try-catch phrase:
        # ... would allow referencing options that are neither defined in config.ini nor the class
        # by returning None for those.

        # debug compatibility with older versions:
        # if attr == 'dset_train':
        #     return '/home/hornebeh/proj_tf_unet/data/hdf5/trainset.h5'
        # elif attr == 'n_class':
        #     return 2
        # elif attr == 'prefetch_threads':
        #     return 12
        # elif attr == 'prefetch_n':
        #     return 32


        return parse_implicitly_extended(self.config_prox, attr)

    def get_attr_list(self):
        """ Return keys of attributes (both from class attributes and config file)"""
        class_attr_keys = [attr for attr in dir(self)
                            if not callable(getattr(self, attr)) and not attr.startswith("__")
                            and not attr == 'config_prox']
        config_keys = [key for key in OrderedDict(self.config_prox.items())]
        return class_attr_keys + config_keys

    def get_attr_val_list(self):
        """ Return keys and values of attributes (both from class attributes and config file)"""
        class_attr = [(attr, getattr(self, attr)) for attr in dir(self)
                            if not callable(getattr(self, attr)) and not attr.startswith("__")
                            and not attr == 'config_prox']
        config_attr = [(attr, parse_implicitly(self.config_prox[attr]))
                       for attr in OrderedDict(self.config_prox.items())]
        return class_attr + config_attr

    def get_attr_val_dict(self):
        """ Return keys and values of attributes as OrderedDict (both from class attributes and config file)"""
        return OrderedDict(self.get_attr_val_list())


def keystr_from_config(conf_file_path = 'config.ini', section = 'DEFAULT'):
    if conf_file_path is None:
        conf_file_path = 'config.ini'

    # create config_reader (to allow config.attr access)
    config = configparser.ConfigParser()
    config.read(conf_file_path)
    opts = config_decorator(config[section])

    # create string from config attributes
    build_string = (('bn%1.3f_' % (opts.norm_fn_param_decay)).replace('0', '') if opts.norm_fn is not None else '') + \
                   ('bs%i' % opts.batch_size if opts.batch_size > 1 else '') + \
                   ('s' if opts.shuffle else '') + \
                   ('A' if opts.augment else '') + \
                   (('Dp%1.2f' % (1 - opts.keep_prob)) if opts.keep_prob < 1.0 else '') + \
                   (('_Re%i' % opts.resample_n) if opts.resample_n is not None else '') + \
                   (('_AL%i%s%s' % (opts.aleatoric_sample_n,
                                     opts.aleatoric_distr if opts.aleatoric_distr is not None else '',
                                    ('_'+opts.aleatoric_reg) if opts.aleatoric_reg is not None else '' ))
                    if opts.aleatoric_sample_n is not None else '') + \
                   (('_%s' % opts.train_name) if (section=='TRAIN' and opts.train_name is not None) else '')

    # remove dots from floats
    return build_string.replace('.', '')


def keystr_from_opts(opts):
    # create string from config attributes
    build_string = (('bn%1.3f_' % (opts.norm_fn_param_decay)).replace('0', '') if opts.norm_fn is not None else '') + \
                   ('bs%i' % opts.batch_size if opts.batch_size > 1 else '') + \
                   ('s' if opts.shuffle else '') + \
                   ('A' if opts.augment else '') + \
                   (('Dp%1.2f' % (1 - opts.keep_prob)) if opts.keep_prob < 1.0 else '') + \
                   (('_Re%i' % opts.resample_n) if opts.resample_n is not None else '') + \
                   (('_AL%i%s%s' % (opts.aleatoric_sample_n,
                                     opts.aleatoric_distr if opts.aleatoric_distr is not None else '',
                                    ('_'+opts.aleatoric_reg) if opts.aleatoric_reg is not None else '' ))
                    if opts.aleatoric_sample_n is not None else '')

    try:
        if opts.global_step is not None:
            build_string = build_string + '_' + str(opts.global_step)
    except:
        pass

    # remove dots from floats
    return build_string.replace('.', '')
