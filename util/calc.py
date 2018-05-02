# CALC
# =========================
#
# math helper functions (implementations of entropy
#
# Author: Hannes Horneber
# Date: 2018-03-22

import numpy as np
from scipy.stats import entropy
from math import log, e

import timeit

def entropy1(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

# def entropy3(labels, base=None):
#     vc = pd.Series(labels).value_counts(normalize=True, sort=False)
#     base = e if base is None else base
#     return -(vc * np.log(vc) / np.log(base)).sum()

def entropy4(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def entropy_bin_array(label_array, nclass=2, axis=0):
    """ Computes entropy of an array of label distributions with binary labels [0, 1]. """
    n_labels = label_array.shape[axis]
    if n_labels <= 1:
        return 0

    binary_count = np.sum(label_array, axis=0)
    p1 = binary_count / n_labels
    ones = np.ones(binary_count.shape, dtype=np.float32)
    p0 = np.subtract(ones, p1)

    # ent = - p1 * np.log2(p1) - p0 * np.log2(p0)
    ent = np.multiply(-1, np.add(np.multiply(p1, np.log2(p1)), np.multiply(p0, np.log2(p0))))

    return np.nan_to_num(ent)

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


# def softmax(x):
#     e_x = np.exp(x - np.max(x));
#     return e_x / e_x.sum()
#     # """Compute softmax for vector x."""
#     # e_x = np.exp(x - np.max(x))
#     # return e_x / e_x.sum()
#
#
# def arg_softmax(x):
#     np.apply_along_axis()
