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
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )