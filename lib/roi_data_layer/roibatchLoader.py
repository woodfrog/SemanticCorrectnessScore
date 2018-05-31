
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.batch_size = batch_size
    self.data_size = len(self._roidb)
    num_batch = int(np.ceil(len(roidb) / batch_size))

  def __getitem__(self, index):
    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db, self._num_classes)

    # print(blobs)
    # import IPython; IPython.embed()
    # assert(1 == 0)

    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        np.random.shuffle(blobs['gt_boxes'])
        try:
          gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        except:
          print('gt_boxes error')
          import IPython; IPython.embed()

        im_info[0, 0] = data.size(1)
        im_info[0, 1] = data.size(2)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        num_boxes = min(gt_boxes.size(0), self.max_num_box)
        gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]

        # permute trim_data to adapt to downstream processing
        try:
          data = data.transpose(2,3).transpose(1,2).contiguous()
        except:
          import IPython; IPython.embed()
        im_info = im_info.view(3)

        return data.squeeze(0), im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
