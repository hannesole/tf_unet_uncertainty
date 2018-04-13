# TENSORFLOW HELPERS
# =========================
#
# Functions for tensorflow tasks.
#
# Author: Hannes Horneber
# Date: 2018-04-10

import logging
import tensorflow as tf

def initialize_uninitialized(sess):
    " This checks global vars for uninitialized variables and initializes them. Adds ops to the graph! "
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    logging.debug('> initialize_uninitialized. Found %s ' % str(len(not_initialized_vars)))
    if len(not_initialized_vars):
        print('\n[...]\n'.join([str(not_initialized_vars[i].name) for i in [0, len(not_initialized_vars) - 1]]))
        sess.run(tf.variables_initializer(not_initialized_vars))