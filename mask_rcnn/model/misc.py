"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import calc
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

import utils
## IMPORT ALL SINGLE FUNCTIONS OF EACH MODEL MODULE
from mask_rcnn.model.data_formatting import compose_image_meta
from mask_rcnn.model.data_formatting import parse_image_meta
from mask_rcnn.model.data_formatting import parse_image_meta_graph
from mask_rcnn.model.data_formatting import mold_image
from mask_rcnn.model.data_formatting import unmold_image

from mask_rcnn.model.data_generator import load_image_gt
from mask_rcnn.model.data_generator import build_detection_targets
from mask_rcnn.model.data_generator import build_rpn_targets
from mask_rcnn.model.data_generator import generate_random_rois
from mask_rcnn.model.data_generator import data_generator

from mask_rcnn.model.graph_FeaturePyramid import fpn_classifier_graph
from mask_rcnn.model.graph_FeaturePyramid import build_fpn_mask_graph

from mask_rcnn.model.graph_regionProposal import rpn_graph
from mask_rcnn.model.graph_regionProposal import build_rpn_model

from mask_rcnn.model.graph_resnet import identity_block
from mask_rcnn.model.graph_resnet import conv_block
from mask_rcnn.model.graph_resnet import resnet_graph

from mask_rcnn.model.graph_mask_rcnn import MaskRCNN

from mask_rcnn.model.layer_detection import clip_to_window
from mask_rcnn.model.layer_detection import refine_detections
from mask_rcnn.model.layer_detection import DetectionLayer

from mask_rcnn.model.layer_detectionTarget import overlaps_graph
from mask_rcnn.model.layer_detectionTarget import detection_targets_graph
from mask_rcnn.model.layer_detectionTarget import DetectionTargetLayer

from mask_rcnn.model.layer_proposal import apply_box_deltas_graph
from mask_rcnn.model.layer_proposal import clip_boxes_graph
from mask_rcnn.model.layer_proposal import ProposalLayer

from mask_rcnn.model.layer_ROIAlign import log2_graph
from mask_rcnn.model.layer_ROIAlign import PyramidROIAlign

from mask_rcnn.model.losses import smooth_l1_loss
from mask_rcnn.model.losses import rpn_class_loss_graph
from mask_rcnn.model.losses import rpn_bbox_loss_graph
from mask_rcnn.model.losses import mrcnn_class_loss_graph
from mask_rcnn.model.losses import mrcnn_bbox_loss_graph
from mask_rcnn.model.losses import mrcnn_mask_loss_graph

from mask_rcnn.model.misc import trim_zeros_graph
from mask_rcnn.model.misc import batch_pack_graph

from mask_rcnn.model.util import log
from mask_rcnn.model.util import BatchNorm
# END IMPORT SINGLE FUNCTIONS FROM MODEL MODULES

# for importing model modules (obsolete)
# from mask_rcnn.model import data_formatting
# from mask_rcnn.model import data_generator
# from mask_rcnn.model import graph_FeaturePyramid
# from mask_rcnn.model import graph_regionProposal
# from mask_rcnn.model import graph_resnet
# from mask_rcnn.model import graph_mask_rcnn
# from mask_rcnn.model import layer_detection
# from mask_rcnn.model import layer_detectionTarget
# from mask_rcnn.model import layer_proposal
# from mask_rcnn.model import layer_ROIAlign
# from mask_rcnn.model import losses
# from mask_rcnn.model import misc
# from mask_rcnn.model import util


# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
