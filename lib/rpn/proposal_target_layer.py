# _*_ coding:utf-8 _*_
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
#import matplotlib  
#matplotlib.use('Agg') 
DEBUG = False

'''
layer {
  name: 'roi-data'
  type: 'Python'
  
  bottom: 'rpn_rois'
  
  bottom: 'gt_boxes'
  bottom: 'data'

  top: 'rois_p2'
  top: 'rois_p3'

  top: 'labels_p2'
  top: 'labels_p3'

  top: 'bbox_targets_p2'
  top: 'bbox_targets_p3'

  top: 'bbox_inside_weights_p2'
  top: 'bbox_inside_weights_p3'

  top: 'bbox_outside_weights_p2'
  top: 'bbox_outside_weights_p3'

  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': 2"
  }
}

target:

layer {
  name: 'roi-data'
  type: 'Python'
  
  bottom: 'rpn_rois_p2'
  bottom: 'rpn_rois_p3'
  
  bottom: 'gt_boxes'
  bottom: 'data'

  top: 'rois_p2'
  top: 'rois_p3'

  top: 'labels_p2'
  top: 'labels_p3'

  top: 'bbox_targets_p2'
  top: 'bbox_targets_p3'

  top: 'bbox_inside_weights_p2'
  top: 'bbox_inside_weights_p3'

  top: 'bbox_outside_weights_p2'
  top: 'bbox_outside_weights_p3'

  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': 2"
  }
}
'''


class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._batch_rois = 256 #cfg.TRAIN.BATCH_SIZE

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        top[1].reshape(1, 5, 1, 1)

        # labels_1
        top[2].reshape(1, 1, 1, 1)
        # labels_2
        top[3].reshape(1, 1, 1, 1)


        # bbox_targets_1
        top[4].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_targets_2
        top[5].reshape(1, self._num_classes * 4, 1, 1)


        # bbox_inside_weights_1
        top[6].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_inside_weights_2
        top[7].reshape(1, self._num_classes * 4, 1, 1)


        # bbox_outside_weights_1
        top[8].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_outside_weights_2
        top[9].reshape(1, self._num_classes * 4, 1, 1)



    def forward(self, bottom, top):

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

        #branch 1 2
        rois_list = []
        branch_num = 2
        for i in xrange(branch_num):
            rois_list.append(bottom[i].data) # 300 [  0.      70.29284668   0.     105.74542236  49.81745911]

        #--debug
        # for i in rois_list:
        #     print(i[0:2])
        # input()

        gt_boxes = bottom[2].data

        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])

        w = (gt_boxes[:, 2] - gt_boxes[:, 0])
        h = (gt_boxes[:, 3] - gt_boxes[:, 1])
        g_s = w * h

        g_s[g_s <= 0] = 1e-6
        gt_index = g_s.copy()

        #### alter ####
        gt_index_list = []


        gt_index[g_s >= 2000] = 1
        gt_index_list.append(gt_index.copy())

        gt_index[g_s >= 3000] = 2
        gt_index_list.append(gt_index.copy())



        rois_list_res = []
        labels_list = []
        bbox_targets_list = []
        bbox_inside_weights_list = []

        branch_num = 2
        for i in xrange(branch_num):
            gt_index = gt_index_list[i]
            g_index = (gt_index == (i+1))
            num_g = sum(g_index)

            # get gt_bbox
            start = 0
            end_g = num_g
            index_range = range(start, end_g)
            if num_g == 0:
                num_g = 1
                each_gt_box = np.zeros((num_g, 5), dtype=np.float32)
            else:
                each_gt_box = np.zeros((num_g, 5), dtype=np.float32)
                each_gt_box[index_range, :] = gt_boxes[g_index, :]


            zeros = np.zeros((each_gt_box.shape[0], 1), dtype=each_gt_box.dtype)
            rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE
            fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)


            rois_list[i] = np.vstack(
                (rois_list[i], np.hstack((zeros, each_gt_box[:, :-1])))
            )

            labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
                rois_list[i], each_gt_box, fg_rois_per_image,
                rois_per_image, self._num_classes)

            rois_list_res.append(rois)
            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)
            bbox_inside_weights_list.append(bbox_inside_weights)

        #--debug
        #print(each_gt_box)

        im = bottom[3].data

        rois_part1 = rois_list_res[0]
        rois_part2 = rois_list_res[1]

        rois_part1 = rois_part1.reshape((rois_part1.shape[0], rois_part1.shape[1], 1, 1))
        top[0].reshape(*rois_part1.shape)
        top[0].data[...] = rois_part1
        rois_part2 = rois_part2.reshape((rois_part2.shape[0], rois_part2.shape[1], 1, 1))
        top[1].reshape(*rois_part2.shape)
        top[1].data[...] = rois_part2

        # classification labels
        # modified by ywxiong
        labels_1 = labels_list[0]
        labels_2 = labels_list[1]

        labels_1 = labels_1.reshape((labels_1.shape[0], 1, 1, 1))
        top[2].reshape(*labels_1.shape)
        top[2].data[...] = labels_1
        labels_2 = labels_2.reshape((labels_2.shape[0], 1, 1, 1))
        top[3].reshape(*labels_2.shape)
        top[3].data[...] = labels_2

        # bbox_targets
        # modified by ywxiong
        bbox_targets_1 = bbox_targets_list[0]
        bbox_targets_2 = bbox_targets_list[1]

        bbox_targets_1 = bbox_targets_1.reshape((bbox_targets_1.shape[0], bbox_targets_1.shape[1], 1, 1))
        top[4].reshape(*bbox_targets_1.shape)
        top[4].data[...] = bbox_targets_1
        bbox_targets_2 = bbox_targets_2.reshape((bbox_targets_2.shape[0], bbox_targets_2.shape[1], 1, 1))
        top[5].reshape(*bbox_targets_2.shape)
        top[5].data[...] = bbox_targets_2

        # bbox_inside_weights
        # modified by ywxiong

        bbox_inside_weights_1 = bbox_inside_weights_list[0]
        bbox_inside_weights_2 = bbox_inside_weights_list[1]

        bbox_inside_weights_1 = bbox_inside_weights_1.reshape(
            (bbox_inside_weights_1.shape[0], bbox_inside_weights_1.shape[1], 1, 1))
        top[6].reshape(*bbox_inside_weights_1.shape)
        top[6].data[...] = bbox_inside_weights_1
        bbox_inside_weights_2 = bbox_inside_weights_2.reshape(
            (bbox_inside_weights_2.shape[0], bbox_inside_weights_2.shape[1], 1, 1))
        top[7].reshape(*bbox_inside_weights_2.shape)
        top[7].data[...] = bbox_inside_weights_2

        # bbox_outside_weights
        # modified by ywxiong

        bbox_inside_weights_1 = bbox_inside_weights_list[0]
        bbox_inside_weights_2 = bbox_inside_weights_list[1]

        bbox_inside_weights_1 = bbox_inside_weights_1.reshape(
            (bbox_inside_weights_1.shape[0], bbox_inside_weights_1.shape[1], 1, 1))
        top[8].reshape(*bbox_inside_weights_1.shape)
        top[8].data[...] = np.array(bbox_inside_weights_1 > 0).astype(np.float32)
        bbox_inside_weights_2 = bbox_inside_weights_2.reshape(
            (bbox_inside_weights_2.shape[0], bbox_inside_weights_2.shape[1], 1, 1))
        top[9].reshape(*bbox_inside_weights_2.shape)
        top[9].data[...] = np.array(bbox_inside_weights_2 > 0).astype(np.float32)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

###### **************** #######
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    #   for ind in inds:
    #       cls = clss[ind]
    #       start = 4 * cls
    #        end = start + 4
    #        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    #        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    #return bbox_targets, bbox_inside_weights
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds

    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
