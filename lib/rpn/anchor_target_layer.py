# _*_ coding:utf-8 _*_
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

'''
layer {
  name: 'rpn-data'
  type: 'Python'
  bottom: 'rpn_cls_score_p2'
  bottom: 'rpn_cls_score_p3'
  bottom: 'gt_boxes'
  bottom: 'im_info'
#---p2----
  top: 'rpn_labels_p2'
  top: 'rpn_bbox_targets_p2'
  top: 'rpn_bbox_inside_weights_p2'
  top: 'rpn_bbox_outside_weights_p2'


#---p3----
  top: 'rpn_labels_p3'
  top: 'rpn_bbox_targets_p3'
  top: 'rpn_bbox_inside_weights_p3'
  top: 'rpn_bbox_outside_weights_p3'


  python_param {
    module: 'rpn.anchor_target_layer'
    layer: 'AnchorTargetLayer'
    param_str: "'feat_stride': 8,16"
  }
}

'''


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._feat_stride = [int(i) for i in layer_params['feat_stride'].split(',')]
        #self._scales = np.array((1.44, 1.95, 2.625, 3.55, 4.79, 6.47, 8.734, 11.785, 15.9, 21.486, 29.001),dtype=float)
        #self._ratios = [2.44]
        #anchor_scales = layer_params.get('scales', (8, ))
        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 1000)


        height_list = []
        width_list = []

        layer_num = 2

        for i in range(layer_num):
            height, width = bottom[i].data.shape[-2:] ## bottom[0][1][2][3][4].data.shape[-2:]

            height_list.append(height)
            width_list.append(width)

        A = 11

        # labels
        #top[0].reshape(1, 1, A * s)
        # bbox_targets
        #top[1].reshape(1, A * 4, s)
        # bbox_inside_weights
        #top[2].reshape(1, A * 4, s)
        # bbox_outside_weights
        #top[3].reshape(1, A * 4, s)

        # p2
        # labels
        top[0].reshape(1, 1, A * height_list[0], width_list[0])
        # bbox_targets
        top[1].reshape(1, A * 4, height_list[0], width_list[0])
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height_list[0], width_list[0])
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height_list[0], width_list[0])

        # p3
        # labels
        top[4].reshape(1, 1, A * height_list[1], width_list[1])
        # bbox_targets
        top[5].reshape(1, A * 4, height_list[1], width_list[1])
        # bbox_inside_weights
        top[6].reshape(1, A * 4, height_list[1], width_list[1])
        # bbox_outside_weights
        top[7].reshape(1, A * 4, height_list[1], width_list[1])



    def forward(self, bottom, top):
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'


        # map of shape (..., H, W)
        h = []
        w = []

        ### layer_num
        layer_num = 2

        for i in range(layer_num):
            height, width = bottom[i].data.shape[-2:] ## bottom[0]/[1].data.shape[-2:]
            h.append(height)
            w.append(width)

        # GT boxes (x1, y1, x2, y2, label)

        #### layer_num
        gt_boxes = bottom[2].data
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        g_w = (gt_boxes[:,2]-gt_boxes[:,0])
        g_h = (gt_boxes[:,3]-gt_boxes[:,1])
        g_s = g_w * g_h
        g_s[g_s<=0]=1e-6
        gt_index = g_s

        gt_index_list = []


        gt_index[g_s >= 2000] = 1
        gt_index_list.append(gt_index.copy())

        gt_index[g_s >= 2600] = 2
        gt_index_list.append(gt_index.copy())

        ### layer_num
        gt_boxes_list = []
        branch_num = 2
        for i in range(branch_num):
            gt_index = gt_index_list[i]
            g_index = (gt_index == (i + 1))
            gt_boxes_list.append(gt_boxes[g_index, :])

        #----------------------------------------------

        im_info = bottom[3].data[0, :]


        all_anchors_list = []
        inds_inside_list = []
        total_anchors = 0

        feat_strides = self._feat_stride # [8,16]

        label_list = []
        bbox_target_list = []
        bbox_inside_weights_list = []
        bbox_outside_weights_list = []



        for feat_id in range(len(feat_strides)):

            #---------------------------same single---------------------------------
            base_anchors = generate_anchors(feat_stride = feat_strides[feat_id])
            num_anchors = base_anchors.shape[0]
            feat_height = h[feat_id]
            feat_width = w[feat_id]

            shift_x = np.arange(0, feat_width) * feat_strides[feat_id]
            shift_y = np.arange(0, feat_height) * feat_strides[feat_id]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

            A = num_anchors
            K = shifts.shape[0]

            all_anchors = (base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((K * A, 4))
            total_anchors = int(K * A)

            inds_inside = np.where((all_anchors[:, 0] >= -self._allowed_border) &
                                   (all_anchors[:, 1] >= -self._allowed_border) &
                                   (all_anchors[:, 2] < im_info[1] + self._allowed_border) &
                                   (all_anchors[:, 3] < im_info[0] + self._allowed_border))[0]

            anchors = all_anchors[inds_inside, :]

            labels = np.empty((len(inds_inside),), dtype=np.float32)
            labels.fill(-1)
            #---------------------------------------------------------------------

            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            gt_boxes = gt_boxes_list[feat_id]
            if not len(gt_boxes):
                gt_boxes = np.zeros((1, 5), dtype=np.float32)

            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0



            # subsample positive labels if we have too many
            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
                #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))


            ##############################################
            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

            bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
            bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights


            # map up to original set of anchors
            labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

            #-------------------------------------------------------------------------

            # labels
            labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
            labels = labels.reshape((1, 1, A * feat_height, feat_width))

            # bbox_targets
            bbox_targets = bbox_targets \
                .reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
            bbox_targets = bbox_targets.reshape(*bbox_targets.shape)

            # bbox_inside_weights
            bbox_inside_weights = bbox_inside_weights \
                .reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
            bbox_inside_weights = bbox_inside_weights.reshape(*bbox_inside_weights.shape)

            # bbox_outside_weights
            bbox_outside_weights = bbox_outside_weights \
                .reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
            bbox_outside_weights = bbox_outside_weights.reshape(*bbox_outside_weights.shape)

            label_list.append(labels)
            bbox_target_list.append(bbox_targets)
            bbox_inside_weights_list.append(bbox_inside_weights)
            bbox_outside_weights_list.append(bbox_outside_weights)



        ##### p2 #######
        labels_1 = label_list[0]
        top[0].reshape(*labels_1.shape)
        top[0].data[...] = labels_1

        # bbox_targets

        bbox_target_1 = bbox_target_list[0]
        top[1].reshape(*bbox_target_1.shape)
        top[1].data[...] = bbox_target_1

        # bbox_inside_weights

        bbox_inside_weights_1 = bbox_inside_weights_list[0]
        top[2].reshape(*bbox_inside_weights_1.shape)
        top[2].data[...] = bbox_inside_weights_1

        # bbox_outside_weights

        bbox_outside_weights_1 = bbox_outside_weights_list[0]
        top[3].reshape(*bbox_outside_weights_1.shape)
        top[3].data[...] = bbox_outside_weights_1


        ##### p3 #######
        labels_2 = label_list[1]
        top[4].reshape(*labels_2.shape)
        top[4].data[...] = labels_2

        # bbox_targets

        bbox_target_2 = bbox_target_list[1]
        top[5].reshape(*bbox_target_2.shape)
        top[5].data[...] = bbox_target_2

        # bbox_inside_weights

        bbox_inside_weights_2 = bbox_inside_weights_list[1]
        top[6].reshape(*bbox_inside_weights_2.shape)
        top[6].data[...] = bbox_inside_weights_2

        # bbox_outside_weights

        bbox_outside_weights_2 = bbox_outside_weights_list[1]
        top[7].reshape(*bbox_outside_weights_2.shape)
        top[7].data[...] = bbox_outside_weights_2

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """" unmap a subset inds of data into original data of size count """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    if cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        assert cfg.TRAIN.RPN_NORMALIZE_MEANS is not None
        assert cfg.TRAIN.RPN_NORMALIZE_STDS is not None
        targets -= cfg.TRAIN.RPN_NORMALIZE_MEANS
        targets /= cfg.TRAIN.RPN_NORMALIZE_STDS
    return targets
