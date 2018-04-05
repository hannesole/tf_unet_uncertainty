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
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = utils.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis]))
    return result


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            detections_batch = []
            _, _, window, _ = parse_image_meta(image_meta)
            for b in range(self.config.BATCH_SIZE):
                detections = refine_detections(
                    rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
                assert gap >= 0
                if gap > 0:
                    detections = np.pad(
                        detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
                detections_batch.append(detections)

            # Stack detections and cast to float32
            # TODO: track where float64 is introduced
            detections_batch = np.array(detections_batch).astype(np.float32)
            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32)

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)
