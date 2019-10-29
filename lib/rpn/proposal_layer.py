# _*_ coding:utf-8 _*_
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import numpy.random as npr

DEBUG = False

'''
layer {
  name: 'proposal'
  type: 'Python'
    bottom: 'im_info'
    bottom: 'rpn_bbox_pred_res4f'
    bottom: 'rpn_bbox_pred_res5c'
    bottom: 'rpn_cls_prob_reshape_res4f'
    bottom: 'rpn_cls_prob_reshape_res5c'
    
    top: 'rpn_rois'
  
    python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16,16"

  }
}


target:

layer {
  name: 'proposal'
  type: 'Python'
    bottom: 'im_info'
    bottom: 'rpn_bbox_pred_res4f'
    bottom: 'rpn_bbox_pred_res5c'
    bottom: 'rpn_cls_prob_reshape_res4f'
    bottom: 'rpn_cls_prob_reshape_res5c'
    
    top: 'rpn_rois_p2'
    top: 'rpn_rois_p3'
  
    python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16,16"

  }
}

'''



class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """


    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        #### fpn-feat ####
        self._feat_stride = [int(i) for i in layer_params['feat_stride'].split(',')]
        self._min_sizes = 16  
        self._num_anchors = 11
        self._output_score = False


        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

        #generate double branches for different branches
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        cfg_key = str('TRAIN' if self.phase == 0 else 'TEST') # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size = self._min_sizes
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want

        im_info = bottom[0].data[0, :]    # im_info = [160. 213. 0.33333]

        batch_size = bottom[1].data.shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")


        ### easy ignore ###
        cls_prob_dict = {
            'stride16': bottom[4].data[:, self._num_anchors:, :, :],
            'stride8': bottom[3].data[:, self._num_anchors:, :, :],
        }
        bbox_pred_dict = {
            'stride16': bottom[2].data,
            'stride8': bottom[1].data,
        }



        proposal_list = []
        score_list = []
        for s in self._feat_stride:
            stride = int(s)
            sub_anchors = generate_anchors(feat_stride=stride)
            scores = cls_prob_dict['stride' + str(s)]  # ['strdie16/32']
            bbox_deltas = bbox_pred_dict['stride' + str(s)] # ['strdie16/32']


            # 1. Generate proposals from bbox_deltas and shifted anchors
            # use real image size instead of padded feature map sizes

            # height, width = int(im_info[0] / stride), int(im_info[1] / stride)
            height, width = scores.shape[-2:]

            # Enumerate all shifts
            shift_x = np.arange(0, width) * stride
            shift_y = np.arange(0, height) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()

            # Enumerate all shifted anchors:
            #
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = self._num_anchors
            K = shifts.shape[0]
            anchors = sub_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))

            # Transpose and reshape predicted bbox transformations to get them
            # into the same order as the anchors:
            #
            # bbox deltas will be (1, 4 * A, H, W) format
            # transpose to (1, H, W, 4 * A)
            # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
            # in slowest to fastest order

            #### ??? ####
            # 1111 bbox_deltas = _clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
            if cfg_key == 'TRAIN' and cfg.TRAIN.RPN_NORMALIZE_TARGETS:
                bbox_deltas *= cfg.TRAIN.RPN_NORMALIZE_STDS
                bbox_deltas += cfg.TRAIN.RPN_NORMALIZE_MEANS
            # Same story for the scores:
            #
            # scores are (1, A, H, W) format
            # transpose to (1, H, W, A)
            # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)

            #### ??? ####
            # 2222 scores = _clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
            ################## 123 ###################
            # Convert anchors into proposals via bbox transformations

            proposals = bbox_transform_inv(anchors, bbox_deltas)

            # 2. clip predicted boxes to image

            proposals = clip_boxes(proposals, im_info[:2])

            # 3. remove predicted boxes with either height or width < threshold
            # (NOTE: convert min_size to input image scale stored in im_info[2])

            keep = _filter_boxes(proposals, min_size * im_info[2]) 
            proposals = proposals[keep, :]
            scores = scores[keep]

            proposal_list.append(proposals) 
            score_list.append(scores) 





        # proposals = np.vstack(proposal_list) #np.vstack
        # scores = np.vstack(score_list) #same stroy


        ################## 45 ###################

        keep_proposal = []
        for idx in xrange(len(proposal_list)):
            scores = score_list[idx]
            proposals = proposal_list[idx]
            order = scores.ravel().argsort()[::-1]
            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
            proposals = proposals[order, :]
            scores = scores[order]

            ################## 678 ###################
            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            det = np.hstack((proposals, scores)).astype(np.float32)
            keep = nms(det,nms_thresh)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]

            proposals = proposals[keep, :]
            scores = scores[keep]

            #--debug
            # print(proposals.shape)
            # print(scores.shape)

            keep_proposal.append(proposals)

        ################## output ###################
        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0


        batch_inds1 = np.zeros((keep_proposal[0].shape[0], 1), dtype=np.float32)
        blob1 = np.hstack((batch_inds1, keep_proposal[0].astype(np.float32, copy=False)))
        # if is_train:

        batch_inds2 = np.zeros((keep_proposal[1].shape[0], 1), dtype=np.float32)
        blob2 = np.hstack((batch_inds2, keep_proposal[1].astype(np.float32, copy=False)))

        #--2018-10-17 15:59:50
        #generate double branches
        top[0].reshape(*(blob1.shape))
        top[0].data[...] = blob1

        top[1].reshape(*(blob2.shape))
        top[1].data[...] = blob2


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """ Remove all boxes with any side smaller than min_size """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
def _clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor
